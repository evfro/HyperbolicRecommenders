import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .preprocessing import combine_samples, generate_evaluation_data
from .sampler import sample_row_wise
from .random import random_seeds


def npz_to_csr(data):
    assert data['format'].item().decode() == 'csr'
    matrix = csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape = data['shape'],
    )
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    return matrix

def read_urm_data(data_dir, batched=False):
    if batched:
        files = ["URM_train.npz", "URM_validation.npz", "URM_train_all.npz", "URM_test.npz"]
    else:
        files = ["URM_train.npz", "URM_validation.npz", "URM_test.npz", "URM_test_negative.npz"]
    return [npz_to_csr(np.load(os.path.join(data_dir, file))) for file in files]   

def generate_urm_experiment_data(data_dir, n_valid_samples=None, seed=None):
    '''
    Datareader for URM data from the Troubling Analysis paper.
    '''    
    train_mat_val, valid_mat, test_mat, neg_mat = read_urm_data(data_dir)
    
    assert train_mat_val.shape == valid_mat.shape == test_mat.shape == neg_mat.shape
    assert valid_mat.nnz == valid_mat.shape[0] # 1 item per user
    assert test_mat.nnz == test_mat.shape[0] # 1 item per user
    n_samples = neg_mat.getnnz(axis=1)
    assert (n_samples[0] == n_samples).all() # fixed number of items per user
    
    if n_valid_samples is None:
        n_valid_samples = n_samples[0]
    
    random_state = np.random.RandomState(seed)
    valid_data = generate_evaluation_data(train_mat_val, valid_mat, n_valid_samples, random_state)
    test_data = combine_samples(
        holdout = dict(zip(*test_mat.nonzero())),
        unobserved = {i: row.indices for i, row in enumerate(neg_mat)},
        random_state = random_state
    )
    train_mat_test = train_mat_val + valid_mat
    return train_mat_val, valid_data, train_mat_test, test_data