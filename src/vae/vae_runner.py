from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import vstack

import os
import sys
import importlib.util

from .vae_utils import CurvatureOptimizer
from ..recvae import validate, get_data, data_folder_path
from ..datasets import observations_loader, UserBatchDataset
from ..datareader import read_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VaeDataset:
    def __init__(self, batch_size: int, in_dim: int, img_dims: Optional[Tuple[int, ...]]) -> None:
        self.batch_size = batch_size
        self._in_dim = in_dim
        self._img_dims = img_dims

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    @property
    def img_dims(self) -> Optional[Tuple[int, ...]]:
        return self._img_dims

    @property
    def in_dim(self) -> int:
        return self._in_dim

    def metrics(self, x_mb_: torch.Tensor, mode: str = "train") -> Dict[str, float]:
        return {}

class HypVaeDataset(VaeDataset):
    def __init__(self, batch_size: int, batch_size_eval: int) -> None:
        super().__init__(batch_size, in_dim=50, img_dims=None)
        self.n_items = 0
    def create_loaders(self, data_dir, data_pack, data_name, 
                       batch_size, batch_size_eval, part='train') -> Tuple[DataLoader, DataLoader]:
        
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        
        if data_pack == "recvae":
            data_ = get_data(data_folder_path(data_dir, data_pack, data_name))
        elif data_pack == "urm":
            data_ = read_data(data_dir, data_pack, data_name)
            
        train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data_
    
        if part == "test":
            train_data = vstack([train_data, valid_in_data + valid_out_data])
            valid_in_data = test_in_data
            valid_out_data = test_out_data
            
        n_items = train_data.shape[1]
        self.n_items = n_items
        self.valid_data = valid_out_data
        
        train_loader = observations_loader(
            observations = train_data,
            batch_size = self.batch_size,
            shuffle = True, # return user batches in random order
            data_factory = UserBatchDataset,
            sparse_batch = True  # can use .to_dense on a batch for calculations
        )

        test_loader = observations_loader(
            # data for generating predictions
            observations = valid_in_data,
            batch_size = self.batch_size_eval,
            shuffle = False,
            data_factory = UserBatchDataset,
            sparse_batch = True,
        )
                
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")        


class Trainer:
    def __init__(self, model, total_anneal_steps, anneal_cap, fixed_curvature):
        self.model = model
        self.epoch = 0
        
        self.total_anneal_steps = total_anneal_steps 
        self.anneal_cap = anneal_cap
        self.fixed_curvature = fixed_curvature
        
    def epoch(self):
        return self.epoch
    
    def ncurvature_param_cond(self, key: str) -> bool:
        return "nradius" in key or "curvature" in key

    def pcurvature_param_cond(self, key: str) -> bool:
        return "pradius" in key

    def build_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        net_params = [
            v for key, v in self.model.named_parameters()
            if not self.ncurvature_param_cond(key) and not self.pcurvature_param_cond(key)
        ]
        neg_curv_params = [v for key, v in self.model.named_parameters() if self.ncurvature_param_cond(key)]
        pos_curv_params = [v for key, v in self.model.named_parameters() if self.pcurvature_param_cond(key)]
        curv_params = neg_curv_params + pos_curv_params
        
        net_optimizer = torch.optim.Adam(net_params, lr=learning_rate)
        if not self.fixed_curvature and not curv_params:
            warnings.warn("Fixed curvature disabled, but found no curvature parameters. Did you mean to set "
                          "fixed=True, or not?")
        if not pos_curv_params:
            c_opt_pos = None
        else:
            c_opt_pos = torch.optim.SGD(pos_curv_params, lr=5e-4)

        if not neg_curv_params:
            c_opt_neg = None
        else:
            c_opt_neg = torch.optim.SGD(neg_curv_params, lr=1e-3)

        def condition() -> bool:
            return (not self.fixed_curvature) and (self.epoch >= 10) 

        return CurvatureOptimizer(net_optimizer, neg=c_opt_neg, 
                                  pos=c_opt_pos, should_do_curvature_step=condition)
    
    def train(self, optimizer, train_loader, update_count):
        # Turn on training mode
        N = len(train_loader.dataset)
        self.model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to_dense()
            batch = batch.to(device)
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            batch_stats, _ = self.model.train_step(optimizer, batch, anneal)

            update_count += 1

        ###print radius value
        if not self.fixed_curvature:
            neg_curv_params = [v for key, v in self.model.named_parameters() if self.ncurvature_param_cond(key)]
            print("radius = {}, c = {}".format(neg_curv_params[0].item(), 
                                               1 / (neg_curv_params[0].item() ** 2))) 
        
        self.epoch += 1
        return update_count
        
    def evaluate(self, valid_loader, valid_data, topk=[20, 100], show_progress=False):
        # Turn on evaluation mode
        self.model.eval()
        return validate(self.model, valid_loader, valid_data,
                               topk=topk, show_progress=show_progress, variational=True)

