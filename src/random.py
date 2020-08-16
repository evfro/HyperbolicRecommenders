import numpy as np
import torch

def check_random_state(random_state):
    """
    Handle seed or random state as input.
    Provide consistent output.
    """
    if random_state is None:
        return np.random
    if isinstance(random_state, (np.integer, int)):
        return np.random.RandomState(random_state)
    return random_state


def random_seeds(size, entropy=None):
    return np.random.SeedSequence(entropy).generate_state(size)


def seed_generator(seed):
    rs = np.random.RandomState(seed)
    while True:
        new_seed = yield rs.randint(np.iinfo('i4').max)
        if new_seed is not None:
            rs = np.random.RandomState(new_seed)

            
def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)