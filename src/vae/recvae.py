import os
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import torch

def read_data(data_dir, data_pack, data_name, part):
    data_folder = os.path.join(data_dir, data_pack, data_name)
    filepath = os.path.join(data_folder, part + '.csv.gz')
    return pd.read_csv(filepath)

def get_train_data(tp):
    rows, cols = tp['uid'], tp['sid']
    data = csr_matrix(
        (np.ones_like(rows), (rows, cols)),
        dtype='float64'
    )
    return data

def get_tr_te_data(tp_tr, tp_te, n_items):
    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = csr_matrix(
        (np.ones_like(rows_tr), (rows_tr, cols_tr)),
        dtype='float64', shape=(end_idx - start_idx + 1, n_items)
    )
    data_te = csr_matrix(
        (np.ones_like(rows_te), (rows_te, cols_te)),
        dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


bn = np # no real reason to use bottleneck over numpy for argpartition

def NDCG_binary_at_k_batch(idx_topk, heldout_batch):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users, k = idx_topk.shape
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(idx_topk, heldout_batch):
    batch_users, k = idx_topk.shape
    X_true_binary = (heldout_batch > 0).toarray()
    X_pred_binary = np.zeros_like(X_true_binary, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk] = True
    
    tmp = np.logical_and(X_true_binary, X_pred_binary).sum(axis=1).astype('f4')
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def validate(model, loader, data_te, topk=[100], show_progress=False, report_coverage=True):
    scores = defaultdict(list)
    coverage_set = defaultdict(set)
    if show_progress:
        loader = tqdm(loader)
    
    for i, batch in enumerate(loader):
        dense_batch = batch.to_dense()
        with torch.no_grad():
            predictions = model(dense_batch)[2]
        
        # exclude examples from training and validation (if any)
        predictions[tuple(batch.coalesce().indices())] = -np.inf
        indices = torch.topk(predictions, max(topk), axis=1)[1].cpu().numpy()
        
        batch_size = batch.size()[0]
        idx = i * batch_size
        test_batch = data_te[idx:idx+batch_size]
        for k in topk:
            scores[f'recall@{k}'].append(Recall_at_k_batch(indices[:, :k], test_batch))
            scores[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(indices[:, :k], test_batch))
            if report_coverage:
                coverage_set[f'cov@{k}'].update(np.unique(indices[:, :k]))
    
    results = {metric:np.concatenate(score).mean() for metric, score in scores.items()}
    if coverage_set:
        n_items = batch.size()[1]
        results.update({metric: len(inds)/n_items for metric, inds in coverage_set.items()})
    return results