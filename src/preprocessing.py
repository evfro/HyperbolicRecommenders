import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .sampler import sample_row_wise
from .random import check_random_state, random_seeds

def reindex(raw_data, index, filter_invalid=True):
    if isinstance(index, pd.Index):
        index = [index]
    
    new_index = {}
    for ix in index:
        assert isinstance(ix, pd.Index)
        assert ix.name is not None
        assert ix.name in raw_data.columns
        reindexed = ix.reindex(raw_data[ix.name].values)[1]
        if reindexed is None:
            # reindex returns None if index is identical
            # have to manually create new index
            reindexed = np.arange(raw_data.shape[0])
        new_index[ix.name] = reindexed
    
    new_data = raw_data.assign(**new_index)
    
    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join(map('{} == -1'.format, new_index.keys()))
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]
    
    return new_data

    
def matrix_from_observations(
        data,
        userid='userid',
        itemid='itemid',
        user_index=None,
        item_index=None,
        feedback=None,
        preserve_order=False,
        shape=None,
        dtype=None
    ):
    '''
    Encodes pandas dataframe into sparse matrix. If index is not provided,
    returns new index mapping, which optionally preserves order of original data.
    Automatically removes incosnistent data not present in the provided index.
    '''
    if (user_index is None) or (item_index is None):
        useridx, user_index = pd.factorize(data[userid], sort=preserve_order)
        itemidx, item_index = pd.factorize(data[itemid], sort=preserve_order)
        user_index.name = userid
        item_index.name = itemid
    else:
        data = reindex(data, (user_index, item_index), filter_invalid=True)
        useridx = data[user_index.name].values
        itemidx = data[item_index.name].values
        if shape is None:
            shape = (len(user_index), len(item_index))

    if feedback is None:
        values = np.ones_like(itemidx, dtype=dtype)
    else:
        values = data[feedback].values

    matrix = csr_matrix((values, (useridx, itemidx)), dtype=dtype, shape=shape)
    return matrix, user_index, item_index


def combine_samples(holdout, unobserved, userid='userid', itemid='itemid', random_state=None):
    users = []
    items = []
    labels = []
    random_state = check_random_state(random_state)
    for user, item in holdout.items():
        n_samples = len(unobserved[user]) + 1
        users.extend([user]*n_samples)
        # insert holdout item at random position to avoid metric intrusion
        # e.g., when predictor always returns 0 as a correct item index 
        # while we place holdout item at position 0 by default
        holdout_pos = random_state.choice(n_samples)
        items.extend(np.insert(unobserved[user], holdout_pos, item))
        # assign labels accordingly
        lbls = [0] * n_samples
        lbls[holdout_pos] = 1
        labels.extend(lbls)
    combined = np.concatenate([users, items, labels]).reshape(-1, 3, order='F')
    return combined


def generate_evaluation_data(train_mat, test_mat, n_samples, random_state):
    n_users, n_items = train_mat.shape
    # generating negative samples that will be added to validation data
    observed = train_mat + test_mat
    seed_seq = random_seeds(n_users, random_state.choice(np.iinfo('i4').max))
    sampled = sample_row_wise(observed.indptr, observed.indices, n_items, n_samples, seed_seq)
    return combine_samples(
        holdout = dict(zip(*test_mat.nonzero())),
        unobserved = {i: row for i, row in enumerate(sampled)},
        random_state = random_state
    )


def generate_experiment_data(observations, holdout, n_negative, *, preserve_order=False, seed=None):
    # convert data into sparse matrix with regular index
    matrix, *new_index = matrix_from_observations(observations, preserve_order=preserve_order)
    # sample negative items for evaluation - must not include holdout
    random_state = np.random.RandomState(seed)
    holdout_mat, *_ = matrix_from_observations(holdout, user_index=new_index[0], item_index=new_index[1])
    evaluation_data = generate_evaluation_data(matrix, holdout_mat, n_negative, random_state)
    return matrix, evaluation_data, new_index
