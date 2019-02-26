import numpy as np
import numba as nb

def just_sum(x2):
    return np.sum(x2, axis=1)

@nb.jit('double[:](double[:, :])', nopython=True)
def nb_just_sum(x2):
    return np.sum(x2, axis=1)

x2=np.random.random((2048,2048))
