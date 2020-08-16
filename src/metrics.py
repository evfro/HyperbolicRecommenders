import numpy as np
import torch


def ndcg_metric(hit_index):
    '''
    In the case of a single-item holdout ndcg is equivalent to dcg,
    as the normalizing factor for a single withheld item is exactly 1.
    '''
    return np.reciprocal(np.log2(hit_index + 2.0))

def arhr_metric(hit_index):
    return np.reciprocal(hit_index + 1.0)

def metrics(predictions, labels, *, top_k=10, true_label=1.0, coverage_set=None):
    _, predicted_items = torch.topk(predictions, top_k)    
    if coverage_set is not None:
        coverage_set.update(predicted_items.tolist())
    
    (hit_index,) = torch.where(labels[predicted_items] == true_label)
    try:
        hit_index = hit_index.item() # we expect only 1 item here
    except ValueError: # empty index or more then 1 item (which is incorrect)
        return 0, 0.0, 0.0 # hits, arhr, ndcg
    
    hits = 1
    arhr = arhr_metric(hit_index)
    ndcg = ndcg_metric(hit_index)
    return hits, arhr, ndcg