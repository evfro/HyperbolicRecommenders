import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import geoopt

from src.batchmodels import HyperbolicAutoEncoder, SimpleAutoEncoder, PureSVD, MobiusAutoEncoder
from src.batchrunner import train, evaluate, report_metrics
from src.datareader import read_data
from src.datasets import observations_loader, UserBatchDataset
from src.random import random_seeds, fix_torch_seed

assert torch.cuda.is_available()

#in our experiments, we have used wandb framework to run experiments
#entity = ...
#project = ...
# import wandb
# wandb.init(entity=entity, project=project)

####################PARAMETERS####################

parser = argparse.ArgumentParser()

parser.add_argument("--datapack", type=str, required=True, choices=["persdiff", "urm"])
parser.add_argument("--dataname", type=str, required=True) # depends on choice of data pack
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--test_negative_samples", type=int, default=999)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--hidden_dim_factor", type=int, default=2)
parser.add_argument("--num_encoders", type=int, default=1)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--step_size", type=int, default=7)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--loss", type=str, default="mse", choices=["mse", "bce"])
parser.add_argument("--show-progress", default=False, action='store_true')
parser.add_argument("--activation", default="no", choices=["tanh", "relu", "no"])
parser.add_argument("--model", default="hyplinear", choices=["hyplinear", "mobius", "linear"])
parser.add_argument("--data_dir", default="./data/")
parser.add_argument("--no-coverage", default=False, action='store_true')
# wandb compatibility
parser.add_argument("--bias", type=str, default="True")
parser.add_argument("--masked_loss", type=str, default="False")
parser.add_argument("--scheduler_on", type=str, default="True")
parser.add_argument("--last_layer_activation", type=str, default="True")

args = parser.parse_args()
###############INITIALIZATION###############

# data description
userid = "userid"
itemid = "itemid"
feedback = None

# randomization control
seeds = random_seeds(6, args.seed)
rand_seed_val, rand_seed_test = seeds[:2]
runner_seed_val, runner_seed_test = seeds[2:4]
sampler_seed_val, sampler_seed_test = seeds[4:]
fix_torch_seed(args.seed)

train_mat_val, valid_data, *unused_test_data = read_data(
    args.data_dir,
    args.datapack,
    args.dataname,
    n_negative_samples=args.test_negative_samples,
    preserve_order=False,
    seed_val = rand_seed_val,
    seed_test = rand_seed_test
)

train_loader = observations_loader(
    observations = train_mat_val,
    batch_size = args.batch_size,
    shuffle = True,
    data_factory = UserBatchDataset,
    sparse_batch = True  # can use .to_dense on a batch for calculations
)

infer_loader = observations_loader(
    observations = train_mat_val,
    batch_size = 1,
    shuffle = False,
    data_factory = UserBatchDataset,
    sparse_batch = True,

)

eval_gr = pd.DataFrame(valid_data).groupby(0, sort=False)
eval_data = dict(
    items_data={uid: torch.cuda.LongTensor(gr.values) for uid, gr in eval_gr[1]},
    label_data={uid: torch.cuda.LongTensor(gr.values) for uid, gr in eval_gr[2]}
)


######################MODEL#######################

# wandb compatibility
bias = (args.bias == "True")
masked_loss = (args.masked_loss =="True")
scheduler_on = (args.scheduler_on == "True")
last_layer_activation = (args.last_layer_activation =="True")

autoencoder_config = dict(
    num_items = train_loader.dataset.num_items,
    latent_dim = args.embedding_dim,
    hidden_dim = args.embedding_dim // args.hidden_dim_factor,
    num_encoders = args.num_encoders,
    activation = args.activation,
    last_layer_activation = True, # <== due to bug all previous computations were made with True, hardcoding it for now 
    bias = True # <== due to bug all previous computations were made with True, hardcoding it for now 
)

if args.model == "linear":
    model = SimpleAutoEncoder(**autoencoder_config).cuda()
elif args.model == "hyplinear":
    model = HyperbolicAutoEncoder(c=args.c, **autoencoder_config).cuda()
elif args.model == "mobius":
    model = MobiusAutoEncoder(c=args.c, **autoencoder_config).cuda()
else:
    raise ValueError('Unrecognized model type')


criterions = {
    "mse": nn.MSELoss(reduction='mean'),
    "bce": nn.BCEWithLogitsLoss()
}

criterion = criterions[args.loss].cuda()

optimizers = {
    "mobius": geoopt.optim.RiemannianAdam,
    "hyplinear": torch.optim.Adam,
    "linear": torch.optim.Adam,    
}
optimizer = optimizers[args.model](
    model.parameters(),
    lr = args.learning_rate    
)

scheduler = None
if scheduler_on:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )



#####################EXPERIMENT######################
show_progress = args.show_progress

for epoch in range(args.epochs):
    losses = train(train_loader, model, optimizer, criterion, 
                   masked_loss=masked_loss, show_progress=show_progress)
    scores = evaluate(infer_loader,eval_data, 
                      model, show_progress=show_progress)
    scores.update({'loss': np.mean(losses)})
#     wandb.log(scores)    
    report_metrics(scores, epoch)
