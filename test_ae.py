import os

import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from src.utils import parse_config
from scipy.sparse import vstack

# +
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
from src.recvae import validate, get_data, data_folder_path
# -

#in our experiments, we have used wandb framework to run experiments
#entity = ...
#project = ...
import wandb
wandb.init(entity=entity, project=project)

# +
parser = argparse.ArgumentParser()
parser.add_argument("--sweepid", type=str) #sweep id
parser.add_argument("--part", type=str, default="train") #train or test
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--batch_size_eval", type=int, default=2000)

parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--step_size", type=int, default=7)
parser.add_argument("--scheduler", default=True, action='store_true')

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-coverage", default=False, action='store_true')
parser.add_argument("--masked_loss", type=str, default="False")
parser.add_argument("--scheduler_on", type=str, default="True")
parser.add_argument("--last_layer_activation", type=str, default="True")

parser.add_argument("--show_progress", default=False, action='store_true')
args = parser.parse_args()
# -

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

api = wandb.Api()
sweep = api.sweep("{}/{}/{}".format(entity, project, args.sweepid))
config = parse_config(sweep.best_run().json_config)

epochs = config['epochs']
data_name = config['dataname']
data_pack = config['datapack']
batch_size = config['batch_size']
if "activation" in config.keys():
    activation = config['activation']
else:
    activation = "no"
hidden_dim_factor = config['hidden_dim_factor']
loss = config['loss']
masked_loss = (config['masked_loss'] == "True")
num_encoders = config['num_encoders']
model = config['model']
embedding_dim = config['embedding_dim']
learning_rate = config['learning_rate']
if data_name in ["pinterest", "ml1m"]:
    test_negative_samples = config['test_negative_samples']
scheduler_on = (config['scheduler_on'] == "True")
if 'c' in config.keys():
    c = config['c']
data_dir = args.data_dir    

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

# +
if data_name in ["netflix", "ml20m"]:
    if data_pack == "recvae":
         data_ = get_data(data_folder_path(data_dir, data_pack, data_name))
    else:
        data_  = read_data(data_dir, data_pack, data_name)
elif data_name in ["pinterest", "ml1m"]:
    data_  = read_data(
        data_dir,
        data_pack,
        data_name,
        n_negative_samples=test_negative_samples,
        preserve_order=False,
        seed_val = rand_seed_val,
        seed_test = rand_seed_test
    )
if data_name in ["netflix", "ml20m"]:
    train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data_
    if args.part == "train":
        train_mat = train_data
        test_data = valid_out_data
        final_test_data = test_out_data
    else:
        train_mat = vstack([train_data, valid_in_data + valid_out_data])
        
elif data_name in ["pinterest", "ml1m"]:
    train_mat_val, valid_data, train_mat_test, test_data = data_
    
    if args.part == "train":
        train_mat = train_mat_val
        test_data = valid_data
        final_test_data = test_data
    else:
        train_mat = train_mat_test

if data_name in ["pinterest", "ml1m"]:
    train_loader = observations_loader(
        observations = train_mat,
        batch_size = batch_size,
        shuffle = True,
        data_factory = UserBatchDataset,
        sparse_batch = True  # can use .to_dense on a batch for calculations
    )
    infer_loader = observations_loader(
        observations = train_mat,
        batch_size = 1,
        shuffle = False,
        data_factory = UserBatchDataset,
        sparse_batch = True
    )

    eval_gr = pd.DataFrame(test_data).groupby(0, sort=False)
    eval_data = dict(
        items_data={uid: torch.cuda.LongTensor(gr.values) for uid, gr in eval_gr[1]},
        label_data={uid: torch.cuda.LongTensor(gr.values) for uid, gr in eval_gr[2]}
    )
    
elif data_name in ["netflix", "ml20m"]:
    train_loader = observations_loader(
        observations = train_mat,
        batch_size = batch_size,
        shuffle = True, # return user batches in random order
        data_factory = UserBatchDataset,
        sparse_batch = True  # can use .to_dense on a batch for calculations
    )

    infer_loader = observations_loader(
        # data for generating predictions
        observations = test_in_data,
        batch_size = args.batch_size_eval,
        shuffle = False,
        data_factory = UserBatchDataset,
        sparse_batch = True,
    )
    eval_data = test_out_data
# -

print("LOADED DATA")

# +
autoencoder_config = dict(
    num_items = train_loader.dataset.num_items,
    latent_dim = embedding_dim,
    hidden_dim = embedding_dim // hidden_dim_factor,
    num_encoders = num_encoders,
    activation = activation,
    last_layer_activation = True, # <== due to bug all previous computations were made with True, hardcoding it for now 
    bias = True # <== due to bug all previous computations were made with True, hardcoding it for now 
)

if model == "linear":
    model = SimpleAutoEncoder(**autoencoder_config).cuda()
elif model == "hyplinear":
    model = HyperbolicAutoEncoder(c=c, **autoencoder_config).cuda()
elif model == "mobius":
    model = MobiusAutoEncoder(c=c, **autoencoder_config).cuda()
else:
    raise ValueError('Unrecognized model type')
# -

print("CREATED MODEL")
print(model)

# +
###############TRAINING##############
criterions = {
    "mse": nn.MSELoss(reduction='mean'),
    "bce": nn.BCEWithLogitsLoss()
}

criterion = criterions[loss].cuda()

optimizers = {
    "mobius": geoopt.optim.RiemannianAdam,
    "hyplinear": torch.optim.Adam,
    "linear": torch.optim.Adam,    
}
optimizer = optimizers[config['model']](
    model.parameters(),
    lr = learning_rate    
)
scheduler = None
if scheduler_on:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    
print("STARTING TRAINING")
# -

c_ = c if "c" in config.keys() else "euc"
save_dir = "checkpoints/{}_{}_{}_{}.pth".format(data_name, config['model'], loss, c_)

# +
metric_key = "ndcg@100" if data_name in ["netflix", "ml20m"] else "ndcg@10"

show_progress = args.show_progress
ndcg = -np.inf
for epoch in range(epochs):
    losses = train(train_loader, model, optimizer, criterion, 
                   masked_loss=masked_loss, show_progress=show_progress)
    
    if data_name in ["netflix", "ml20m"]:
        scores = validate(
            model, infer_loader, eval_data,
            show_progress=show_progress, topk=[10,20,50,100])
    else:
        scores = evaluate(infer_loader,eval_data, 
                          model, show_progress=show_progress, top_k=[1, 5, 10])
    scores.update({'loss': np.mean(losses)})
    if scores[metric_key] > ndcg:
        ndcg = scores[metric_key]
        torch.save(model.state_dict(), save_dir)
    
    wandb.log(scores)    
    report_metrics(scores, epoch)
# -

if args.part == "train":
    #load best model from validation
    print("LOADED MODEL WEIGHTS")
    model.load_state_dict(torch.load(save_dir))
    if data_name in ["netflix", "ml20m"]:
        scores = validate(
            model, infer_loader, final_test_data,
            show_progress=show_progress, topk=[10,20,50,100])
    else:
        scores = evaluate(infer_loader, final_test_data, 
                          model, show_progress=show_progress, 
                          top_k=[1, 5, 10])
    scores.update({'loss': np.mean(losses)})
    wandb.log(scores)    
    report_metrics(scores, epoch)
