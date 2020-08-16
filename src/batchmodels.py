import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

import torch
import geoopt
import torch.nn as nn
from hyptorch.nn import HypLinear

from .mobius_linear import MobiusLinear

class PureSVD:
    '''
    Compatible with torch dataloader during evaluation phase.
    '''
    def __init__(self, rank=10, randomized=False):
        self.randomized = randomized
        self.rank = rank
        self.item_factors = None
        self.train_matrix = None
        self.name = 'PureSVD'

    def fit(self, matrix):
        self.train_matrix = matrix
        if self.randomized:
            *_, vt = randomized_svd(self.train_matrix, self.rank)
            self.item_factors = torch.cuda.FloatTensor(vt.T)
        else:
            *_, vt = svds(self.train_matrix, k=self.rank, return_singular_vectors='vh')
            self.item_factors = torch.cuda.FloatTensor(np.ascontiguousarray(vt[::-1, :].T))

    def __call__(self, batch, *, rank=None):
        factors = self.item_factors
        if rank is not None:
            assert rank <= self.item_factors.shape[1]
            factors = self.item_factors[:, :rank]
        
        if batch.ndim == 1:
            return (factors @ (factors.T @ batch.view(-1, 1))).squeeze()
        return (batch @ factors) @ factors.T
    
    def train(self):
        pass
    
    def eval(self):
        pass


def select_layer(activation):
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "no":
        return None
    raise ValueError

def clone_layers(num_layers, layer_factory, activation, in_dim, out_dim, bias, **kwargs):
    layers = []
    for i in range(num_layers):
        layers.extend([
            layer_factory(in_dim if i==0 else out_dim, out_dim, bias=bias, **kwargs),
            select_layer(activation)
        ])
    return layers

def compose_autoencoder(input_dim, latent_dim, hidden_dim, num_encoders, activation, last_layer_activation, bias, layer_factory, **kwargs):    
    encoder_modules = clone_layers(1, layer_factory, activation, input_dim, latent_dim, bias, **kwargs)
    encoder_modules += clone_layers(num_encoders-1, layer_factory, activation, latent_dim, hidden_dim, bias, **kwargs)            

    decoder_modules = clone_layers(int(num_encoders>1), layer_factory, activation, hidden_dim, latent_dim, bias, **kwargs)
    decoder_modules += clone_layers(1, layer_factory, activation if last_layer_activation else "no", latent_dim, input_dim, bias, **kwargs)

    encode = nn.Sequential(*[mod for mod in encoder_modules if mod is not None])
    decode = nn.Sequential(*[mod for mod in decoder_modules if mod is not None])
    return encode, decode


class SimpleAutoEncoder(nn.Module):
    def __init__(
        self,
        num_items,
        *,
        latent_dim = 64,
        hidden_dim = 32,
        num_encoders = 1,
        activation = "no",
        last_layer_activation = True,
        bias = True
    ):
        super().__init__()
        layer_factory = nn.Linear
        self.encode, self.decode = compose_autoencoder(
            num_items, latent_dim, hidden_dim, num_encoders, activation, last_layer_activation, bias, layer_factory
        )

    def forward(self, x):
        return self.decode(self.encode(x))


class HyperbolicAutoEncoder(nn.Module):
    def __init__(self,
        num_items,
        *,
        latent_dim=64,
        hidden_dim=32,
        num_encoders=1,
        activation="no",
        last_layer_activation=True,
        bias=True,
        c=1.0
    ):
        super().__init__()
        layer_factory = HypLinear
        self.encode, self.decode = compose_autoencoder(
            num_items, latent_dim, hidden_dim, num_encoders, activation, last_layer_activation, bias, layer_factory, c=c
        )

    def forward(self, x):
        return self.decode(self.encode(x))
    
    
class MobiusAutoEncoder(nn.Module):
    def __init__(self, num_items, *, latent_dim=64, hidden_dim=32, num_encoders=1, c=0.5, activation="no", last_layer_activation=True, bias=True):
        super().__init__()
        
        encoder_modules = [
            MobiusLinear(num_items, latent_dim,
                bias=bias, c=c, nonlin=select_layer(activation))
        ]
        for i in range(num_encoders-1):
            encoder_modules.append(
                MobiusLinear(latent_dim if i == 0 else hidden_dim,
                    hidden_dim, bias=bias, c=c, nonlin=select_layer(activation))
            )
        
        decoder_modules = []
        if num_encoders > 1:
            decoder_modules.append(
                MobiusLinear(hidden_dim, latent_dim,
                        bias=bias, c=c, nonlin=select_layer(activation))
            )
        decoder_modules.append(
            MobiusLinear(latent_dim, num_items, bias=bias, c=c,
                nonlin=select_layer(activation) if last_layer_activation else None)
        )
        
        self.encode = nn.Sequential(*[mod for mod in encoder_modules if mod is not None])
        self.decode = nn.Sequential(*[mod for mod in decoder_modules if mod is not None])

    def forward(self, x):
        return self.decode(self.encode(x))