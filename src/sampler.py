from random import randrange
from random import seed as set_seed
import numpy as np
from numba import njit, prange
from .random import check_random_state


@njit(parallel=True)
def sample_element_wise(indptr, indices, n_cols, n_samples, seed_seq):
    """
    For every nnz entry of a CSR matrix, samples indices not present
    in its corresponding row.
    """
    result = np.empty((indptr[-1], n_samples), dtype=indices.dtype)
    for i in prange(len(indptr) - 1):
        head = indptr[i]
        tail = indptr[i + 1]

        seen_inds = indices[head:tail]
        state = prime_sampler_state(n_cols, seen_inds)
        remaining = n_cols - len(seen_inds)
        set_seed(seed_seq[i])
        for j in range(head, tail):
            sampler_state = state.copy()
            sample_unseen(n_samples, sampler_state, remaining, result[j, :])
    return result


@njit(parallel=True)
def sample_row_wise(indptr, indices, n_cols, n_samples, seed_seq):
    """
    For every row of a CSR matrix, samples indices not present in this row.
    """
    n_rows = len(indptr) - 1
    result = np.empty((n_rows, n_samples), dtype=indices.dtype)
    for i in prange(n_rows):
        head = indptr[i]
        tail = indptr[i + 1]
        seen_inds = indices[head:tail]
        state = prime_sampler_state(n_cols, seen_inds)
        remaining = n_cols - len(seen_inds)
        set_seed(seed_seq[i])
        sample_unseen(n_samples, state, remaining, result[i, :])
    return result


@njit(fastmath=True)
def sample_unseen(sample_size, sampler_state, remaining, result):
    """
    Sample a desired number of integers from a range (starting from zero)
    excluding black-listed elements defined in sample state. Used with in
    conjunction with `prime_sample_state` method, which initializes state.
    Inspired by Fischer-Yates shuffle.
    """
    # gradually sample from the decreased size range
    for k in range(sample_size):
        i = randrange(remaining)
        result[k] = sampler_state.get(i, i)
        remaining -= 1
        sampler_state[i] = sampler_state.get(remaining, remaining)
        sampler_state.pop(remaining, -1)


@njit(fastmath=True)
def prime_sampler_state(n, exclude):
    """
    Initialize state to be used in `sample_unseen_items`.
    Ensures seen items are never sampled by placing them
    outside of sampling region.
    """
    # initialize typed numba dicts
    state = {n: n}; state.pop(n)
    track = {n: n}; track.pop(n)

    n_pos = n - len(state) - 1
    # reindex excluded items, placing them in the end
    for i, item in enumerate(exclude):
        pos = n_pos - i
        x = track.get(item, item)
        t = state.get(pos, pos)
        state[x] = t
        track[t] = x
        state.pop(pos, n)
        track.pop(item, n)
    return state


def sample_unseen_arr(pool_size, sample_size, exclude, random_state=None):
    """Efficient sampling from a range with exclusion"""
    assert (pool_size - len(exclude)) >= sample_size
    random_state = check_random_state(random_state)
    src = random_state.rand(pool_size)
    np.put(src, exclude, -1)  # will never get to the top
    return np.argpartition(src, -sample_size)[-sample_size:]