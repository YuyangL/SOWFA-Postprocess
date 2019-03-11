import numpy as np
import numba as nb
from Utilities import timer

# def just_sum(x2):
#     return np.sum(x2, axis=1)
#
# @nb.jit('double[:](double[:, :])', nopython=True)
# def nb_just_sum(x2):
#     return np.sum(x2, axis=1)
#
# x2=np.random.random((2048,2048))



@timer
# prange has to run with parallel = True
@nb.jit(parallel = True)
def dumbSumPrange(x2 = np.random.random((4096,4096))):
    x3 = 0.
    for i in nb.prange(x2.shape[0]):
        # val = i
        # print(i, val)
        for j in range(x2.shape[1]):
            x3 += x2[i, j]
    print(x3)
    return x3


@timer
def dumbSum(x2 = np.random.random((4096,4096))):
    x3 = 0.
    for i in range(x2.shape[0]):
        # print(i)
        for j in range(x2.shape[1]):
            x3 += x2[i, j]

    print(x3)
    return x3

for i in range(3):
    x3 = dumbSumPrange(x2 = np.random.random((10000, 10000)))

# for i in range(3):
#     x3 = dumbSum(x2 = np.random.random((4096,4096)))
