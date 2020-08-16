import numpy as np
from scipy.sparse import csr_matrix, diags, eye, block_diag, bmat
from scipy.sparse import issparse
from scipy.sparse.linalg import norm
from sklearn.utils.extmath import randomized_svd
import torch.nn as nn
import geoopt
import torch

from hyptorch.nn import HyperbolicDistanceLayer, ConcatPoincareLayer, HypLinear, HyperbolicMLR, ToPoincare


class PureSVD:
    '''
    Compatible with torch dataloader during evaluation phase.
    '''
    def __init__(self, rank=10):
        self.rank = rank
        self.item_factors = None
        self.train_matrix = None
        self.name = 'PureSVD'

    def fit(self, matrix):
        self.train_matrix = matrix
        *_, self.item_factors = randomized_svd(self.train_matrix, self.rank)
        self.item_factors = self.item_factors.T

    def __call__(self, users, items):
        users = users.cpu().numpy()
        items = items.cpu().numpy()
        item_factors = self.item_factors[items, :self.rank]
        user_factors = self.train_matrix[users].dot(self.item_factors[:, :self.rank])
        predictions = (user_factors * item_factors).sum(axis=1) # according to p = VV^T a
        return torch.Tensor(predictions) # make compatible with loader
    
    def eval(self):
        pass


class LookupEmbedding(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, manifold=geoopt.Euclidean(), _weight=None
    ):

        super(LookupEmbedding, self).__init__()
        if isinstance(embedding_dim, int):
            embedding_dim = (embedding_dim,)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        if _weight is None:
            _weight = torch.Tensor(num_embeddings, *embedding_dim)
            self.weight = geoopt.ManifoldParameter(_weight, manifold=self.manifold)
            self.reset_parameters()
        else:
            assert _weight.shape == (
                num_embeddings,
                *embedding_dim,
            ), "_weight MUST be of shape (num_embeddings, *embedding_dim)"
            self.weight = geoopt.ManifoldParameter(_weight, manifold=self.manifold)

    def reset_parameters(self):
        with torch.no_grad():
            data = self.manifold.random_normal(
                *self.weight.shape, mean=0, std=1e-2
            ).data
            # @TODO: better init
            self.weight.data = data

    def forward(self, input):
        shape = list(input.shape) + list(self.weight.shape[1:])
        shape = tuple(shape)
        return self.weight.index_select(0, input.reshape(-1)).view(shape)


class HyperbolicCF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, c=1):
        super().__init__()

        manifold = geoopt.PoincareBall(c=c)
        self.user_embeddings = LookupEmbedding(
            user_num, embedding_dim, manifold=manifold
        )
        self.item_embeddings = LookupEmbedding(
            item_num, embedding_dim, manifold=manifold
        )

        self.sim_layer = HyperbolicDistanceLayer(c=c)
        self.post_sim = nn.Linear(1, 1)

    def forward(self, user, item):
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)

        sim = -self.sim_layer(user_embedding, item_embedding)
        sim = self.post_sim(sim)
        return sim.view(-1)

    def name(self):
        return "HyperbolicCF"


class BPRMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user, item_i, item_j=None):
        user = self.user_embeddings(user)
        item_i = self.item_embeddings(item_i)
        prediction_i = (user * item_i).sum(dim=-1)
        if item_j is None:  # support validation phase
            return prediction_i

        item_j = self.item_embeddings(item_j)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j


# source for NCF code is taken form https://raw.githubusercontent.com/guoyang9/NCF
class NCF(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        factor_num,
        num_layers,
        dropout,
        model,
        GMF_model=None,
        MLP_model=None,
    ):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ["MLP", "GMF"]:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == "NeuMF-pre":
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(
                self.predict_layer.weight, a=1, nonlinearity="sigmoid"
            )

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat(
                [
                    self.GMF_model.predict_layer.weight,
                    self.MLP_model.predict_layer.weight,
                ],
                dim=1,
            )
            precit_bias = (
                self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias
            )

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == "MLP":
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == "GMF":
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == "GMF":
            concat = output_GMF
        elif self.model == "MLP":
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


class CosineDistance(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num, embedding_dim)
        self.initialize()

    def initialize(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user, item):
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)

        normalize = nn.functional.F.normalize
        user_embedding = normalize(user_embedding)
        item_embedding = normalize(item_embedding)

        sim = (user_embedding * item_embedding).sum(1)
        return sim.view(-1)

    def name(self):
        return "CosineDistance"


class HyperbolicNCF(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim=32,
        meta_data=None,
        num_layers=3,
        c=1.0,
        activation="tanh",
        concat="hyp",
        predict="mlr"
    ):

        super(HyperbolicNCF, self).__init__()
        manifold = geoopt.PoincareBall(c=c)

        self.user_embeddings = LookupEmbedding(
            user_num, embedding_dim * (2 ** (num_layers - 1)), manifold=manifold
        )
        self.item_embeddings = LookupEmbedding(
            item_num, embedding_dim * (2 ** (num_layers - 1)), manifold=manifold
        )
        self.is_meta = False
        if meta_data is not None:
            self.is_meta = True
            self.register_buffer("meta", meta_data)

        modules = []
        for i in range(num_layers):
            input_size = embedding_dim * (2 ** (num_layers - i))
            modules.append(HypLinear(input_size, input_size // 2, c=c))
            if activation == "tanh":
                modules.append(nn.Tanh())
            elif activation == "relu":
                modules.append(nn.ReLU())

        self.layers = nn.Sequential(*modules)
        if meta_data is not None:
            assert c == 1.0
            meta_embedding_dim = self.meta.shape[1]
            self.predict_layer = HypLinear(embedding_dim + meta_embedding_dim, 1, c=c)
            self.meta_concat = ConcatPoincareLayer(
                embedding_dim,
                meta_embedding_dim,
                embedding_dim + meta_embedding_dim,
                c=c,
            )
        else:
            if predict == "hyplinear":
                self.predict_layer = HypLinear(embedding_dim, 1, c=c) 
            elif predict == "mlr":
                self.predict_layer = HyperbolicMLR(embedding_dim, 1, c=c) 

        ed = embedding_dim * (2 ** (num_layers - 1))

        self.concat_type = concat
        if self.concat_type == "hyp":
            self.concat = ConcatPoincareLayer(ed, ed, 2 * ed, c=c)

    def forward(self, user, item):
        user_embeddings = self.user_embeddings(user)
        item_embeddings = self.item_embeddings(item)

        if self.concat_type == "hyp":
            interaction = self.concat(user_embeddings, item_embeddings)
        elif self.concat_type == "torch":
            interaction = torch.cat((user_embeddings, item_embeddings), -1)

        output = self.layers(interaction)

        if self.is_meta:
            z = self.meta[item]
            output = self.meta_concat(output, z)

        prediction = self.predict_layer(output)
        return prediction.view(-1)
    
    
class NewHCF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, num_layers, dropout=0.0, c=1.0):
        super(NewHCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        """
        self.user_embedding = nn.Embedding(user_num, embedding_dim)
        self.item_embedding = nn.Embedding(item_num, embedding_dim)

        modules = []
        for i in range(num_layers):
            input_size = embedding_dim // (2 ** i)
            modules.append(nn.Linear(input_size, input_size // 2))
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.ReLU())
        self.layers = nn.Sequential(*modules)

        self.e2p = ToPoincare(c=c)
        self._init_weight_()

        self.sim_layer = HyperbolicDistanceLayer(c=c)
        self.post_sim = nn.Linear(1, 1)
        
    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                    
    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)

        user = self.layers(user_embedding)
        item = self.layers(item_embedding)

        p_user = self.e2p(user)
        p_item = self.e2p(item)

        sim = -self.sim_layer(p_user, p_item)
        sim = self.post_sim(sim)
        return sim.view(-1)        
