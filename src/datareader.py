import os
from src.urm import generate_urm_experiment_data, read_urm_data
from src.persdiff import generate_persdiff_experiment_data


def read_data(data_dir, datapack, dataname, *, n_negative_samples=None, preserve_order=False, seed_val=None, seed_test=None):
    if datapack == "persdiff":
        if n_negative_samples is None:
            n_negative_samples = 999 # paper default
        return generate_persdiff_experiment_data(
            os.path.join(data_dir, datapack, dataname),
            n_samples = n_negative_samples,
            preserve_order = preserve_order,
            seed_val = seed_val,
            seed_test = seed_test
        )

    if datapack == "urm":
        if dataname in ["netflix", "ml20m"]:
            n_test_users = 10000 if dataname == "ml20m" else 40000
            
            train, valid, full_train, test = read_urm_data(
                os.path.join(data_dir, 'troublinganalysis/mvae', dataname),
                batched=True
            )
            # RecVAE style naming
            train_data = train[:-2*n_test_users]
            valid_in_data = full_train[-2*n_test_users:-n_test_users]
            valid_out_data = valid[-2*n_test_users:-n_test_users]
            test_in_data = full_train[-n_test_users:]
            test_out_data = test[-n_test_users:]
            return train_data, valid_in_data, valid_out_data, test_in_data, test_out_data
            
        # the number of negative samples for test is fixed (data is provided),
        # however, we can alter the number of neg. samples for validation            
        if dataname in ["pinterest", "ml1m"]:
            if n_negative_samples is None:
                n_negative_samples = 99 # paper default            
            return generate_urm_experiment_data(
                os.path.join(data_dir, 'troublinganalysis/neumf', dataname),
                n_valid_samples = n_negative_samples,
                seed = seed_val
            )
    
    raise ValueError("Unrecognized datapack or dataname")