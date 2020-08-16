import os

import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from src.vae import vae_utils
from src.vae.vae_models import RecSysVAE
from src.vae.vae_utils import str2bool, CurvatureOptimizer
from src.vae.vae_runner import HypVaeDataset, Trainer
from src.batchrunner import report_metrics
from src.random import random_seeds, fix_torch_seed

# +
#in our experiments, we have used wandb framework to run experiments
#entity = ...
#project = ...
# import wandb
# wandb.init(entity=entity, project=project)
# -

parser = argparse.ArgumentParser()
parser.add_argument("--datapack", type=str, required=True, choices=["recvae", "urm"])
parser.add_argument("--dataname", type=str, required=True) # depends on choice of data pack
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--batch_size_eval", type=int, default=2000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--step_size", type=int, default=7)
parser.add_argument("--scheduler", default=True, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--embedding_dim", type=int, default=600)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--c", type=float, default=0.005)
parser.add_argument('--model', type=str, default="e200", help="Model latent space description.")
parser.add_argument("--fixed_curvature", type=str2bool, default=True, 
                    help="Whether to fix curvatures to (-1, 0, 1).")
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument("--show_progress", default=False, action='store_true')
parser.add_argument("--multilayer", default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

radius = 1.0 / np.sqrt(args.c)
if args.model[0] != "e":
    components = vae_utils.parse_components(args.model, args.fixed_curvature, radius=radius)
else:
    components = vae_utils.parse_components(args.model, args.fixed_curvature)
print(components)

# ##############INITIALIZATION###############

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

data_dir, data_pack, data_name = args.data_dir, args.datapack, args.dataname
dataset = HypVaeDataset(batch_size=args.batch_size, 
                        batch_size_eval=args.batch_size_eval
                       )

print("LOADED DATA")

#train, val, test loaders  
train_loader, valid_loader = dataset.create_loaders(data_dir, data_pack, data_name, 
                                                    batch_size=args.batch_size, 
                                                    batch_size_eval=args.batch_size_eval
                                                   )
###model

ddims = [args.embedding_dim, dataset.n_items]

if args.multilayer:
    ddims = [args.embedding_dim // 2, args.embedding_dim,  dataset.n_items]

model = RecSysVAE(
    decoder_dims=ddims, 
    components=components,
    dataset=dataset, scalar_parametrization=False,
    dropout=args.dropout
).to(device)

print("CREATED MODEL")
print(model)

update_count = 0
N = len(train_loader.dataset)
print("STARTING TRAINING")

trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, 
                 anneal_cap=args.anneal_cap, fixed_curvature=args.fixed_curvature)

optimizer = trainer.build_optimizer(learning_rate=args.learning_rate)

scheduler = None
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

valid_data = dataset.valid_data
try:
    for epoch in range(1, args.epochs + 1):
        update_count = trainer.train(optimizer=optimizer, 
                                     train_loader=train_loader, 
                                     update_count=update_count
                                    )
        if scheduler:
            scheduler.step()
            
        scores = trainer.evaluate(valid_loader, valid_data, topk=[20,100], show_progress=args.show_progress)
        report_metrics(scores, epoch)
#         wandb.log(scores)

except KeyboardInterrupt:
    print('-' * 102)

