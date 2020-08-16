import os
# os.environ["NUMBA_NUM_THREADS"]="18"

import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from src.vae import vae_utils
from src.utils import parse_config
from src.vae.vae_models import RecSysVAE
from src.vae.vae_utils import str2bool, CurvatureOptimizer
from src.vae.vae_runner import HypVaeDataset, Trainer
from src.batchrunner import report_metrics
from src.random import random_seeds, fix_torch_seed

#in our experiments, we have used wandb framework to run experiments
#entity = ...
#project = ...
import wandb
wandb.init(entity=entity, project=project)

parser = argparse.ArgumentParser()
parser.add_argument("--sweepid", type=str) #sweep id
parser.add_argument("--part", type=str, default="train") #train or test
parser.add_argument("--batch_size_eval", type=int, default=2000)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--step_size", type=int, default=7)
parser.add_argument("--scheduler", default=True, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fixed_curvature", type=str2bool, default=True, 
                    help="Whether to fix curvatures to (-1, 0, 1).")
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument("--show_progress", default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

api = wandb.Api()
sweep = api.sweep("{}/{}/{}".format(entity, project, args.sweepid))
config = parse_config(sweep.best_run().json_config)

# +
model_type = config['model']
if model_type[0] != "e":
    c = config['c']
    radius = 1.0 / np.sqrt(c)

epochs = config['epochs']
data_dir = args.data_dir
data_name = config['dataname']
data_pack = config['datapack']
dropout = config['dropout']
batch_size = config['batch_size']
embedding_dim = config['embedding_dim']
learning_rate = config['learning_rate']
# -

if model_type[0] != "e":
    components = vae_utils.parse_components(model_type, args.fixed_curvature, radius=radius)
else:
    components = vae_utils.parse_components(model_type, args.fixed_curvature)
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

dataset = HypVaeDataset(batch_size=batch_size, 
                        batch_size_eval=args.batch_size_eval
                       )

print("LOADED DATA")

#train, val, test loaders  
train_loader, valid_loader = dataset.create_loaders(data_dir, data_pack, data_name, 
                                                    batch_size=batch_size, 
                                                    batch_size_eval=args.batch_size_eval,
                                                    part=args.part
                                                   )
###############MODEL###############

model = RecSysVAE(
    decoder_dims=[embedding_dim, dataset.n_items], 
    components=components,
    dataset=dataset, 
    scalar_parametrization=False,
    dropout=dropout
).to(device)

print("CREATED MODEL")
print(model)

###############TRAINING##############
update_count = 0
N = len(train_loader.dataset)
print("STARTING TRAINING")

trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, 
                 anneal_cap=args.anneal_cap, fixed_curvature=args.fixed_curvature)

optimizer = trainer.build_optimizer(learning_rate=learning_rate)

scheduler = None
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

valid_data = dataset.valid_data
if args.part == "train":
    c_ = "euc" if model_type[0] == "e" else c
    save_dir = "checkpoints/{}_{}_{}.pth".format(data_name, model_type, c_)

n100 = -np.inf
for epoch in range(epochs):
    update_count = trainer.train(optimizer=optimizer, 
                                 train_loader=train_loader, 
                                 update_count=update_count
                                )
    if scheduler:
        scheduler.step()

    scores = trainer.evaluate(valid_loader, valid_data, topk=[20,50,100], show_progress=args.show_progress)
    if scores['ndcg@100'] > n100 and args.part == 'train':
        n100 = scores['ndcg@100']
        torch.save(model.state_dict(), save_dir)

    report_metrics(scores, epoch)
    wandb.log(scores)

if args.part == "train":
    #load best model from validation
    print("LOADED MODEL WEIGHTS")
    model.load_state_dict(torch.load(save_dir))
    
    #new trainer
    trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, 
                 anneal_cap=args.anneal_cap, fixed_curvature=args.fixed_curvature)
    #test loaders
    train_loader, valid_loader = dataset.create_loaders(data_dir, data_pack, data_name, 
                                                    batch_size=batch_size, 
                                                    batch_size_eval=args.batch_size_eval,
                                                    part="train"
                                                   )
    valid_data = dataset.valid_data
    scores = trainer.evaluate(valid_loader, valid_data, topk=[20,50,100], show_progress=args.show_progress)
    print("TEST SCORES")
    report_metrics(scores, epoch + 1)
    wandb.log(scores)
