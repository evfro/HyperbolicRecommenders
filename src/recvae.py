import os
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch

bn = np # no need to use bottleneck over numpy for argpartition on these datasets

def data_folder_path(data_dir, data_pack, data_name):
    '''Used to conform with recvae data reading routines'''
    return os.path.join(data_dir, data_pack, data_name)


# the source below is copied from the RecVae source repository [19.05.2020]
# https://github.com/ilya-shenbin/RecVAE

def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    tp = pd.read_csv(csv_file)
    
    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users, global_indexing=False):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    if global_indexing:
        start_idx = 0
        end_idx = len(unique_uid) - 1
    else:
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def get_data(dataset, global_indexing=False):
    unique_sid = list()
    with open(os.path.join(dataset, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    unique_uid = list()
    with open(os.path.join(dataset, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
            
    n_items = len(unique_sid)
    n_users = len(unique_uid)
    
    train_data = load_train_data(os.path.join(dataset, 'train.csv.gz'), n_items, n_users, global_indexing=global_indexing)
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(dataset, 'validation_tr.csv.gz'),
                                               os.path.join(dataset, 'validation_te.csv.gz'),
                                               n_items, n_users, 
                                               global_indexing=global_indexing)

    test_data_tr, test_data_te = load_tr_te_data(os.path.join(dataset, 'test_tr.csv.gz'),
                                                 os.path.join(dataset, 'test_te.csv.gz'),
                                                 n_items, n_users, 
                                                 global_indexing=global_indexing)
    
    data = train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
    data = (x.astype('float32') for x in data)
    
    return data

def ndcg(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def recall(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()    
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


# ############OUR INTERFACE FOR RECVAE EXPERIMENTS#############

def validate_original(model, loader, data_te, topk=[100], show_progress=False):
    '''
    This is an equivalent adoption of RecVae source for our evaluation pipeline.
    Applies sorting multiple times (which may be inefficient), does not compute coverage.
    '''
    scores = defaultdict(list)
    
    if show_progress:
        loader = tqdm(loader)
    
    for i, batch in enumerate(loader):
        dense_batch = batch.to_dense()
        with torch.no_grad():
            predictions = model(dense_batch)
        
        # exclude examples from training and validation (if any)
        predictions[tuple(batch.coalesce().indices())] = -np.inf
        pred_arr = predictions.cpu().numpy()
        
        batch_size = batch.shape[0]
        idx = i * batch_size
        test_batch = data_te[idx:idx+batch_size]
        for k in topk:
            scores[f'recall@{k}'].append(recall(pred_arr, test_batch, k=k))
            scores[f'ndcg@{k}'].append(ndcg(pred_arr, test_batch, k=k))
    
    results = {metric:np.concatenate(score).mean() for metric, score in scores.items()}
    return results


# equivalent but more economic implementation, which also allows computing coverage;
# for equivalence test, see experiments/MetricIntruserBatchTest.ipynb notebook

def ndcg_economic(predictions, idx_topk, heldout_batch):
    batch_users, k = idx_topk.shape
    # sort indices, as required for ndcg
    top_unsorted = np.take_along_axis(predictions, idx_topk, 1)
    top_sorted_pos = np.argsort(-top_unsorted, axis=1)
    top_sorted_idx = np.take_along_axis(idx_topk, top_sorted_pos, 1)
    
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    
    dcg = (
        heldout_batch[np.arange(batch_users)[:, np.newaxis], top_sorted_idx].toarray() * tp
    ).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return dcg / idcg

def recall_economic(idx_topk, heldout_batch):
    batch_users, k = idx_topk.shape
        
    X_pred_binary = np.zeros(heldout_batch.shape, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk] = True
    
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = np.logical_and(X_true_binary, X_pred_binary).sum(axis=1).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def validate(model, loader, data_te, topk=[100], show_progress=False, report_coverage=True, variational=False):
    scores = defaultdict(list)
    coverage_set = defaultdict(set)
    
    if show_progress:
        loader = tqdm(loader)
    
    for i, batch in enumerate(loader):
        dense_batch = batch.to_dense()
        with torch.no_grad():
            predictions = model(dense_batch)
            
        if variational:
            predictions = predictions[2]
        
        # exclude examples from training and validation (if any)
        predictions[tuple(batch.coalesce().indices())] = -np.inf
        pred_arr = predictions.cpu().numpy()
        # find topk elements positions
        top_idx = np.argpartition(-pred_arr, max(topk), axis=1)
        
        batch_size = batch.shape[0]
        idx = i * batch_size
        test_batch = data_te[idx:idx+batch_size]
                
        for k in topk:
            scores[f'recall@{k}'].append(recall_economic(top_idx[:, :k], test_batch))
            scores[f'ndcg@{k}'].append(ndcg_economic(pred_arr, top_idx[:, :k], test_batch))
            if report_coverage:
                coverage_set[f'cov@{k}'].update(np.unique(top_idx[:, :k]))
    
    results = {metric:np.concatenate(score).mean() for metric, score in scores.items()}
    if coverage_set:
        n_items = batch.size()[1]
        results.update({metric: len(inds)/n_items for metric, inds in coverage_set.items()})
    return results
