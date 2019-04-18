import numpy as np
import numba as nb
from Utilities import timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def just_sum(x2):
#     return np.sum(x2, axis=1)
#
# @nb.jit('double[:](double[:, :])', nopython=True)
# def nb_just_sum(x2):
#     return np.sum(x2, axis=1)
#
# x2=np.random.random((2048,2048))



# @timer
# # prange has to run with parallel = True
# @nb.jit(parallel = True)
# def dumbSumPrange(x2 = np.random.random((4096,4096))):
#     x3 = 0.
#     for i in nb.prange(x2.shape[0]):
#         # val = i
#         # print(i, val)
#         for j in range(x2.shape[1]):
#             x3 += x2[i, j]
#     print(x3)
#     return x3
#
#
# @timer
# def dumbSum(x2 = np.random.random((4096,4096))):
#     x3 = 0.
#     for i in range(x2.shape[0]):
#         # print(i)
#         for j in range(x2.shape[1]):
#             x3 += x2[i, j]
#
#     print(x3)
#     return x3
#
# for i in range(3):
#     x3 = dumbSumPrange(x2 = np.random.random((10000, 10000)))

# for i in range(3):
#     x3 = dumbSum(x2 = np.random.random((4096,4096)))



R, Y = np.meshgrid(np.arange(0, 500, 0.5), np.arange(0, 40, 0.5))
z = 0.1*np.abs(np.sin(R/40)*np.sin(Y/6))

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.pbaspect = (2., 0.6, 0.25)
ax.plot_surface(R, Y, z)
