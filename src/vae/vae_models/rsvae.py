from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .vae import ModelVAE
from ..vae_runner import VaeDataset
from ..components import Component

class RecSysVAE(ModelVAE):
    def __init__(self, decoder_dims: List[int], components: List[Component], dataset: VaeDataset,
                 scalar_parametrization: bool, encoder_dims=None, dropout=0.5) -> None:
        super().__init__(decoder_dims[0], components, dataset, scalar_parametrization)
        if not encoder_dims:
            encoder_dims = decoder_dims[::-1]     
        self.encoder_dims, self.decoder_dims = encoder_dims, decoder_dims #encoder and decoder dims      
        self.decoder_dims.insert(0, self.total_z_dim)
        # 1 hidden layer encoder
        self.en_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.encoder_dims[:-1], self.encoder_dims[1:])])
        
        self.de_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.decoder_dims[:-1], self.decoder_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.in_dim = dataset.n_items

    def init_weights(self):
        for layer in self.en_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.de_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    

    def encode(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        bs, dim = x.shape
        assert dim == self.in_dim
        
        x = x.view(bs, self.in_dim)
        
        x = F.normalize(x)
        x = self.drop(x)
        
        for i, layer in enumerate(self.en_layers):
            x = layer(x)
            x = torch.tanh(x)
        
        return x.view(bs, -1)

    
    def decode(self, concat_z: Tensor) -> Tensor:
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)
        
        for i, layer in enumerate(self.de_layers[:-1]):
            concat_z = layer(concat_z)
            concat_z = torch.tanh(concat_z)
            
        concat_z = self.de_layers[-1](concat_z)

        concat_z = concat_z.view(-1, bs, self.in_dim)  # flatten
        return concat_z.squeeze(dim=0)  # in case we're not doing LL estimation
    
    
    def train_step(self, optimizer: torch.optim.Optimizer, x_mb: Tensor, beta: float):
        optimizer.zero_grad()

        x_mb = x_mb.to(self.device)
        reparametrized, concat_z, x_mb_ = self(x_mb)
        assert x_mb_.shape == x_mb.shape
        batch_stats = self.compute_batch_stats(x_mb, x_mb_, reparametrized, likelihood_n=0, beta=beta)

        loss = -batch_stats.elbo / x_mb.shape[0] # Maximize elbo instead of minimizing it.
        assert torch.isfinite(loss).all()
        loss.backward()
        c_params = [v for k, v in self.named_parameters() if "curvature" in k]
        if c_params:  # TODO: Look into this, possibly disable it.
            torch.nn.utils.clip_grad_norm_(c_params, max_norm=1.0, norm_type=2)  # Enable grad clip?
        optimizer.step()   
        return batch_stats.convert_to_float(), (reparametrized, concat_z, x_mb_)    

