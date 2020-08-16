import os
import numpy as np
import pandas as pd

from .preprocessing import generate_experiment_data

def read_data(data_dir, part):
    dtypes = {'userid': 'intp', 'itemid': 'intp'}
    filepath = os.path.join(data_dir, part + '.csv.gz')
    return pd.read_csv(filepath, dtype=dtypes)

def generate_persdiff_experiment_data(data_dir, *, n_samples=999, preserve_order=False, seed_val=0, seed_test=99):
    # validation data
    train_data = read_data(data_dir, 'train')
    holdout_val = read_data(data_dir, 'valid')
    train_mat_val, valid_data, _ = generate_experiment_data(
        train_data,
        holdout_val,
        preserve_order = preserve_order,
        n_negative = n_samples,
        seed = seed_val
    )
    # test data
    full_data = train_data.append(holdout_val, ignore_index=True)
    holdout_test = read_data(data_dir, 'test')
    train_mat_test, test_data, _ = generate_experiment_data(
        full_data,
        holdout_test,
        n_negative = n_samples,
        preserve_order = preserve_order,
        seed = seed_test
    )
    return train_mat_val, valid_data, train_mat_test, test_data