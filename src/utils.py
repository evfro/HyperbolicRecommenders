from io import BytesIO
from urllib.request import urlopen
import numpy as np
import pandas as pd
import json

def read_npz_form_url(url, allow_pickle=False):
    """Read numpy's .npz file directly from source url."""
    with urlopen(url) as response:
        file_handle = BytesIO(response.read())
        return np.load(file_handle, allow_pickle=allow_pickle)
        
def generate_data(
    n_users=1000,
    n_items=100,
    max_items=25,
    min_items=5,
    userid="userid",
    itemid="itemid",
    seed=None,
):
    rs = np.random.RandomState(seed)
    data = pd.DataFrame.from_records(
        [
            (user, item)
            for user in range(n_users)
            for item in rs.choice(
                n_items, size=rs.randint(min_items, max_items + 1), replace=False
            )
        ],
        columns=[userid, itemid],
    )
    return data


def split_contiguous(idx):
    idx_diff = np.diff(idx)
    assert (idx_diff >= 0).all(), 'Index must be sorted.'
    split_pos = np.where(idx_diff)[0] + 1
    return np.concatenate(([0], split_pos, [len(idx)]))

class ArgsBase:
    def __repr__(self):
        max_len = max([len(attr) for attr in dir(self) if not attr.startswith('_')])
        return '=== configuration ===\n\n' + '\n'.join([
            f'{attr:<{max_len+1}} = {getattr(self, attr)}'
            for attr in dir(self) if not attr.startswith('_')
        ])

def parse_config(json_config):
    config = json.loads(json_config)
    keys = config.keys()
    values = [v['value'] for v in list(config.values())]
    return dict(zip(keys, values))    