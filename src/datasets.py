import numpy as np
from scipy.sparse import isspmatrix_csr
import torch
import torch.utils.data as td

from .sampler import sample_element_wise
from .random import random_seeds, seed_generator


def observations_loader(
        observations = None,
        data = None,
        n_neg_samples = 0,
        batch_size = 256,
        sampler = None,
        shuffle = True,
        seed = None,
        data_factory = None,
        **kwargs, # to feed into data_factory
    ):
    if (observations is None) and (data is None):
        raise ValueError('Either observations or data must be set!')
    
    if data_factory is None:
        data_factory = ObservationsDataset
    
    dataset = data_factory(
        observations = observations,
        data = data,
        negative_samples = n_neg_samples,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed,
        **kwargs
    )

    if sampler is None:
        sampler = SamplerWithReset
    data_sampler = sampler(dataset)
    
    return td.DataLoader(
        dataset,
        batch_size = None, # disable torch collation_fn
        batch_sampler = None, # disable collation_fn
        shuffle = False, # handled via dataset
        sampler = data_sampler,
    )


################### DATASET FACTORY SECTION ###################


class Interactions(td.Dataset):
    '''
    Wrapper for interaction tensors (users, items, feedback, etc.).
    Provides convenient indexing and manipulation routines.
    Handles None, if present in interactions, which allows to 
    later update the dataset with newly generated samples. 
    '''
    def __init__(self, *interactions):
        self.interactions = interactions
        self.device = self.interactions[0].device
        self._verify()

    def _verify(self):
        assert all(
            (
                isinstance(tensor, torch.Tensor) and
                tensor.size(0) == len(self) and
                tensor.device == self.device
            )
            for tensor in self.interactions
            if tensor is not None
        )
        
    def __getitem__(self, idx):
        return tuple(
            tensor[idx] for tensor in self.interactions if tensor is not None
        )

    def __len__(self):
        return self.interactions[0].size(0)
    
    def add_interactions(self, others):
        for tensor, other in zip(self.interactions, others):
            if tensor is not None:
                assert tensor.type() == other.type()
        
        return Interactions(*[
            torch.cat((tensor, other))
            for tensor, other in zip(self.interactions, others)
            if tensor is not None
        ])
    
    @staticmethod
    def empty_like(other):
        assert isinstance(other, Interactions)
        return Interactions(*[
            torch.empty(
                torch.Size((0,)),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device
            ) for tensor in other.interactions
        ])
    
    def expand_like(self, other, pos=-1):
        if pos < 0: # support indexing from the end
            pos += len(self.interactions)
        assert 0 <= pos < len(self.interactions)
        
        n_repeat, rem = divmod(other.size(0), len(self))
        assert rem == 0, 'Tensor size is incompatible'
        
        return Interactions(*[
            tensor.expand(n_repeat, -1).reshape(-1) if i!=pos else other
            for i, tensor in enumerate(self.interactions)
        ])

    def shuffle(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randperm(
            len(self),
            dtype = torch.long,
            device = self.device
        )
        return Interactions(*[
            tensor[idx] if tensor is not None else None for tensor in self.interactions
        ])
    
    def to_sparse(self, dims=None):
        unqs, rows = self.interactions[0].unique(return_inverse=True)
        cols = self.interactions[1]
        vals = self.interactions[2]
        
        size = list(dims) if dims else None
        if (size is None) or (size == [None, None]):
            size = [len(unqs), cols.max()+1]
        elif size[0] is None:
            size[0] = len(unqs)
        elif size[1] is None:
            size[1] = cols.max() + 1
        
        inds = torch.cat([rows, cols]).view(2, -1)
        return torch.cuda.sparse.FloatTensor(inds, vals, torch.Size(size))

    def __add__(self, other):
        # disable default ConcatDataset which is inapplicable here
        raise NotImplementedError


class ObservationsDatasetBase:
    '''
    Boilerplate for working with observations data. Should be subclassed.
    '''
    def __init__(self, observations, data, negative_samples, batch_size, shuffle, seed):
        self.observations = observations
        self.data = data
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._sampler_state = seed_generator(seed)
        self._shuffle_state = seed_generator(seed)
        self.interactions = None
        self._interactions = self.initialize_interactions()
        self.reset()
    
    def initialize_interactions(self):
        raise NotImplementedError
    
    def __len__(self):
        n_full_batches, rem = divmod(len(self.interactions), self.batch_size)
        return n_full_batches + int(rem>0)
    
    def _get_batch(self, item):
        batch_size = self.batch_size
        num_samples = len(self.interactions)
        idx = item * batch_size
        if idx >= num_samples:
            raise IndexError
        return self.interactions[idx:idx+batch_size] # also handles odd size batches
    
    def __getitem__(self, item):
        return self._get_batch(item)
    
    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError
        
    def __iadd__(self, other):
        raise NotImplementedError

    def sample_negatives(self, random_state):
        raise NotImplementedError
                    
    def reset_random_state(self):
        self._sampler_state.send(self.seed)
        self._shuffle_state.send(self.seed)
        
    def reset(self):
        self.interactions = self._interactions
        # function below should avoid rewriting self._interactions
        if self.negative_samples > 0:
            self.interactions = self + self.sample_negatives(next(self._sampler_state))
        if self.shuffle:
            self.interactions = self.interactions.shuffle(next(self._shuffle_state))        
    
    @staticmethod
    def read_observations(data=None, observations=None):
        if data is not None: # for validation data
            assert isinstance(data, np.ndarray)
            useridx = data[:, 0]
            itemidx = data[:, 1]
            labels = data[:, 2]
        else: # for train data
            assert isspmatrix_csr(observations)
            if not observations.has_sorted_indices:
                observations.sort_indices()
            useridx, itemidx = observations.nonzero()
            labels = [1] * len(useridx)
        return useridx, itemidx, labels


class ObservationsDataset(ObservationsDatasetBase):
    '''
    Simple dataset class for collaborative filtering task. Implements vectorized
    numpy-style batch indexing, avoids element-wise operations. All operations,
    except negative sampling, are performed on GPU, avoiding costly IO between
    CPU and GPU. Suitable for large yet very sparse data. Implementation is motivated
    by discussion and examples at https://github.com/pytorch/pytorch/issues/21645.
    '''
    def initialize_interactions(self):
        useridx, itemidx, labels = self.read_observations(self.data, self.observations)
        return Interactions(
            torch.cuda.LongTensor(useridx),
            torch.cuda.LongTensor(itemidx),
            torch.cuda.FloatTensor(labels)
        )

    def sample_negatives(self, random_state):
        n_users, n_items = self.observations.shape
        seed_seq = random_seeds(n_users, random_state)
        items = sample_element_wise(
            indptr = self.observations.indptr,
            indices = self.observations.indices,
            n_cols = n_items,
            n_samples = self.negative_samples,
            seed_seq = seed_seq
        )
        users = np.broadcast_to(
            np.repeat(
                np.arange(n_users),
                np.diff(self.observations.indptr)
            )[:, np.newaxis],
            items.shape
        )
        labels = [0] * len(items.flat)
        
        neg_users = torch.cuda.LongTensor(users.ravel())
        neg_items = torch.cuda.LongTensor(items.ravel())
        neg_labels = torch.cuda.FloatTensor(labels)
        return neg_users, neg_items, neg_labels
    
    def __add__(self, other):
        return self.interactions.add_interactions(other)

    
class UserBatchDataset(ObservationsDatasetBase):
    '''
    Generates batch data by user. Batch size defines the number of users in a batch.
    '''
    def __init__(self, observations, data, negative_samples, batch_size, shuffle, seed, sparse_batch=True):
        if data is not None:
            raise NotImplementedErorr
        if negative_samples:
            raise ValueError('Negative sampling is not performed for batch models')
        
        assert (observations.getnnz(axis=1) > 0).all(), "There must be no empty rows in the training matrix"
        
        self.shuffle_idx = None # initialized via reset
        super().__init__(observations, data, negative_samples, batch_size, shuffle, seed)
        self.sparse_batch = sparse_batch
        self.num_items = observations.shape[1]

            
    def initialize_interactions(self):
        self.observations.sort_indices()
        useridx, itemidx, labels = self.read_observations(None, self.observations)
        self.index_splits = self.observations.indptr
        return Interactions(
            torch.cuda.LongTensor(useridx),
            torch.cuda.LongTensor(itemidx),
            torch.cuda.FloatTensor(labels)
        )
    
    def __len__(self):
        n_full_batches, rem = divmod(len(self.index_splits)-1, self.batch_size)
        return n_full_batches + int(rem>0)
    
    def _get_batch(self, item):
        batch_size = self.batch_size
        idx = item * batch_size
        if idx >= len(self.shuffle_idx):
            raise IndexError
        batch = Interactions.empty_like(self.interactions)
        for uid in self.shuffle_idx[idx:idx+batch_size]:
            batch_start = self.index_splits[uid]
            batch_end = self.index_splits[uid+1]
            batch = batch.add_interactions(self.interactions[batch_start:batch_end])
            
        if self.sparse_batch:
            return self.batch_to_sparse(batch)
        return batch.interactions
    
    def batch_to_sparse(self, batch):
        return batch.to_sparse(dims=[None, self.num_items])
    
    def reset(self):
        self.interactions = self._interactions
        if self.shuffle:
            random_state = np.random.RandomState(next(self._shuffle_state))
            self.shuffle_idx = random_state.permutation(len(self.index_splits)-1)
        else:
            self.shuffle_idx = np.arange(len(self.index_splits)-1)
            
    

class BPRDataset(ObservationsDataset):
    '''
    Simple dataset class for Bayesian Personalized Ranking models.
    '''
    def initialize_interactions(self):
        useridx, itemidx, _ = self.read_observations(self.data, self.observations)        
        return Interactions(
            torch.cuda.LongTensor(useridx),
            torch.cuda.LongTensor(itemidx),
            None # no negatives at initialization            
        )
                            
    def sample_negatives(self, random_state):
        n_users, n_items = self.observations.shape
        seed_seq = random_seeds(n_users, random_state)
        items = sample_element_wise(
            indptr = self.observations.indptr,
            indices = self.observations.indices,
            n_cols = n_items,
            n_samples = self.negative_samples,
            seed_seq = seed_seq
        )        
        return torch.cuda.LongTensor(items.ravel())
    
    def __add__(self, other):
        return self.interactions.expand_like(other)


class SamplerWithReset(td.SequentialSampler):
    def __iter__(self):
        self.data_source.reset()
        return super().__iter__()
