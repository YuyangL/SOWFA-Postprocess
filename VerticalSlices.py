from FieldData import FieldData
import numpy as np
from math import sqrt

# fields = FieldData('U')
#
# U = fields.readFieldsData()['U']
#
# ccx, ccy, ccz, cc = fields.readCellCenterCoordinates()
#
# meshSize, cellSizeMin, ccx3D, ccy3D, ccz3D = fields.getMeshInfo(ccx, ccy, ccz)
#
# Uslice, ccSlice, sliceDim = fields.createSliceData(U, (1500, 1500, 0), normalVector = (0.5, -sqrt(3)/2., 0))
#
# UmagSlice = np.zeros((Uslice.shape[0], 1))
# for i, row in enumerate(Uslice):
#     UmagSlice[i] = np.sqrt(row[0]**2 + row[1]**2 + row[2]**2)
#
# UmagSliceMesh = UmagSlice.reshape((sliceDim[2], sliceDim[0]))
# X = ccSlice[:, 0].reshape((sliceDim[2], sliceDim[0]))
# Z = ccSlice[:, 2].reshape((sliceDim[2], sliceDim[0]))
# Y = ccSlice[:, 1].reshape((sliceDim[2], sliceDim[0]))


# './ABL_N_H/Slices/20000.9038025/U_alongWind_Slice.raw'
# 'I:/SOWFA Data/ALM_N_H/Slices/20500.9078025/U_alongWind_Slice.raw'
data = np.genfromtxt('/media/yluan/Toshiba External Drive/ALM_N_H/Slices/22000.0558025/U_hubHeight_Slice.raw', skip_header = 2)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Mesh size in x
valOld = x[0]
for i, val in enumerate(x[1:]):
    if val < valOld:
        print(val, valOld)
        meshSizeXY = i + 1
        break

    valOld = val

X, Y, Z = x.reshape((-1, meshSizeXY)), y.reshape((-1, meshSizeXY)), z.reshape((-1, meshSizeXY))

u, v, w = data[:, 3], data[:, 4], data[:, 5]

# U
UmagSlice = np.zeros((data.shape[0], 1))
for i, row in enumerate(data):
    if np.nan in row:
        print(row)
    UmagSlice[i] = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2)

# uvMagSlice = np.zeros((data.shape[0], 1))
# for i, row in enumerate(data):
#     if np.nan in row:
#         print(row)
#     uvMagSlice[i] = np.sqrt((row[3] - 8*np.cos(np.pi/6))**2 + (row[4] - 8*np.sin(np.pi/6))**2)

UmagSliceMesh = UmagSlice.reshape((-1, meshSizeXY))
# uvMagSliceMesh = uvMagSlice.reshape((-1, meshSizeXY))
uMesh, vMesh, wMesh = u.reshape((-1, meshSizeXY)), v.reshape((-1, meshSizeXY)), w.reshape((-1, meshSizeXY))



# # FFt
# uvFft, wFft = np.fft.fft2(uvMagSliceMesh), np.fft.fft2(wMesh)
# # uFft = npfft.fft(uMesh)
# uFft = uvFft
#
# nX, nY = uFft.shape[1], uFft.shape[0]
# freqX, freqY = np.fft.fftfreq(nX, d = 10.), np.fft.fftfreq(nX, d = 10.)
#
# freqX, freqY = np.fft.fftshift(freqX), np.fft.fftshift(freqY)




# # now we can initialize some arrays to hold the wavenumber co-ordinates of each cell
# kx_array = np.zeros(uFft.shape)
# ky_array = np.zeros(uFft.shape)
#
# # before we can calculate the wavenumbers we need to know the total length of the spatial
# # domain data in x and y. This assumes that the spatial domain units are metres and
# # will result in wavenumber domain units of radians / metre.
# x_length = 3000.
# y_length = 3000.
#
# # now the loops to calculate the wavenumbers
# for row in range(uFft.shape[0]):
#
#   for column in range(uFft.shape[1]):
#
#     kx_array[row][column] = ( 2.0 * np.pi * freqs[column] ) / x_length
#     ky_array[row][column] = ( 2.0 * np.pi * freqs[row] ) / y_length


# # Is that right?
# # Shift freqs all to non-negative
# kX, kY = 2*np.pi*(freqX - freqX.min()), 2*np.pi*(freqY - freqY.min())
#
# krOld = 0
# E, kr = np.zeros((uFft.shape[0], 1)), np.zeros((uFft.shape[0], 1))
# for i in range(uFft.shape[0]):
#     kr[i] = np.sqrt(kX[i]**2 + kY[i]**2)
#     dk = abs(krOld - kr[i])
#     eii = float(uFft[i, i]*np.conj(uFft[i, i]))
#     E[i] = eii/2.
#
#     krOld = kr[i]



from PlottingTool import PlotSurfaceSlices3D
myplot = PlotSurfaceSlices3D(X, Y, Z, UmagSliceMesh, name = 'surf', figDir = './', xLim = (0, 3000), yLim = (0, 3000), zLim = (0, 1000), viewAngles = (20, -100))

myplot.initializeFigure()

myplot.plotFigure()

myplot.finalizeFigure()





# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import griddata
# from scipy import interpolate
#
#
# # plt.loglog(kr, E)
#
#
#
#
#
# # Refinement
# # gridX, gridY, gridZ = np.mgrid[X.min():X.max():200j, Y.min():Y.max():200j, Z.min():Z.max():150j]
# # Uinterp = griddata(ccSlice, UmagSlice, (gridX, gridY, gridZ))
#
# # gridX, gridZ = np.mgrid[X.min():X.max():200j, Z.min():Z.max():150j]
# # tck = interpolate.bisplrep(X, Z, UmagSliceMesh, s=0)
# # Uinterp = interpolate.bisplev(gridX[:, 0], gridZ[0, :], tck)
#
# colorDim = UmagSliceMesh
# colorMin, colorMax = colorDim.min(), colorDim.max()
# norm = mpl.colors.Normalize(colorMin, colorMax)
# cmap = plt.cm.ScalarMappable(norm = norm, cmap = 'plasma')
# cmap.set_array([])
# fColors = cmap.to_rgba(colorDim)
#
#
# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 1.2, 0.6, 1]))
# # cstride/rstride refers to use value of every n cols/rows for facecolors
# plot = ax.plot_surface(X, Y, Z, cstride = 1, rstride = 1, facecolors = fColors, vmin = colorMin, vmax = colorMax,
#                        shade = False)
# plt.colorbar(cmap, extend = 'both')
#
#
# ax.set_xlim(0, 3000)
# ax.set_ylim(0, 3000)
# ax.set_zlim(0, 1000)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(20, -100)
# plt.tight_layout()
# plt.show()
# #
# # # from PlottingTool import plotSlices3D
# # # plotSlices3D([UmagSliceMesh], X, Z, [0])
