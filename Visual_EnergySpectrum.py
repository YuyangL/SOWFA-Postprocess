# from PostProcess_FieldData import FieldData
import numpy as np
import os
import matplotlib.pyplot as plt
import PostProcess_EnergySpectrum
import time

t0 = time.time()

case = 'ABL_N_L/Slices'

x2D, y2D, z2D, U2D, u2D, v2D, w2D = PostProcess_EnergySpectrum.readSliceRawData('U_hubHeight_Slice.raw')

E, Evert, Kr = PostProcess_EnergySpectrum.getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = (10., 10.))

t1 = time.time()

ticToc = t1 - t0

plt.figure('horizontal')
plt.loglog(Kr, E)
plt.loglog(Kr, Kr**(-5/3.)*(9e-3)**(2/3.)*1.5)
plt.tight_layout()

plt.figure('vertical')
plt.loglog(Kr, Evert)
plt.loglog(Kr, Kr**(-5/3.)*(9e-3)**(2/3.)*1.5)
plt.tight_layout()


# def readSliceRawData(field, case = 'ABL_N_H/Slices', caseDir = './', time = None, skipHeader = 2):
#     timePath = caseDir + '/' + case + '/'
#     if time is None:
#         time = os.listdir(timePath)
#         try:
#             time.remove('Result')
#         except:
#             pass
#
#         time = time[0]
#
#     fieldFullPath = timePath + str(time) + '/' + field
#     data = np.genfromtxt(fieldFullPath, skip_header = skipHeader)
#
#     # 1D array
#     x, y, z = data[:, 0], data[:, 1], data[:, 2]
#
#     # Mesh size in x
#     valOld = x[0]
#     for i, val in enumerate(x[1:]):
#         if val < valOld:
#             nPtX = i + 1
#             break
#
#         valOld = val
#
#     x2D, y2D, z2D = x.reshape((-1, nPtX)), y.reshape((-1, nPtX)), z.reshape((-1, nPtX))
#
#     if data.shape[1] == 6:
#         u, v, w = data[:, 3], data[:, 4], data[:, 5]
#         scalarField = np.zeros((data.shape[0], 1))
#         for i, row in enumerate(data):
#             scalarField[i] = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2)
#
#     else:
#         u, v, w = np.zeros((data.shape[1], 1)),np.zeros((data.shape[1], 1)), np.zeros((data.shape[1], 1))
#         scalarField = data[:, 3]
#
#     u2D, v2D, w2D = u.reshape((-1, nPtX)), v.reshape((-1, nPtX)), w.reshape((-1, nPtX))
#     scalarField2D = scalarField.reshape((-1, nPtX))
#
#     print('\nSlice raw data read')
#     return x2D, y2D, z2D, scalarField2D, u2D, v2D, w2D
#
#
#
# x2D, y2D, z2D, U2D, u2D, v2D, w2D = readSliceRawData('U_hubHeight_Slice.raw')
#
#
# def getEnergySpectrum(u2D, v2D, w2D):
#     uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
#
#     uResFft = np.fft.fft2(uRes2D)
#     vResFft = np.fft.fft2(vRes2D)
#     wResFft = np.fft.fft2(wRes2D)
#     uResFft, vResFft, wResFft = np.fft.fftshift(uResFft), np.fft.fftshift(vResFft), np.fft.fftshift(wResFft)
#
#     nX, nY = uRes2D.shape[1], uRes2D.shape[0]
#     freqX, freqY = np.fft.fftfreq(nX, d = 10.), np.fft.fftfreq(nY, d = 10.)
#     freqX, freqY = np.fft.fftshift(freqX), np.fft.fftshift(freqY)
#
#     Eii = abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft))
#
#     Kr = np.zeros((len(freqX)*len(freqY), 1))
#     EiiR = np.zeros((len(freqX)*len(freqY), 1))
#     iR = 0
#     for iX in range(len(freqX)):
#         for iY in range(len(freqY)):
#             # Kcurrent = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
#             # if Kcurrent in Kr:
#
#             Kr[iR] = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
#             EiiR[iR] = Eii[iY, iX]
#             iR += 1
#
#     KrSorted = np.sort(Kr, axis = 0)
#     sortIdx = np.argsort(Kr, axis = 0).ravel()
#     EiiSorted = EiiR[sortIdx]
#
#     EiiReduced, Kreduced = [], []
#     i = 0
#     while i < len(KrSorted):
#         EiiRrepeated = EiiSorted[i]
#         anyMatch = False
#         for j in range(i + 1, len(KrSorted)):
#             if KrSorted[j] == KrSorted[i]:
#                 EiiRrepeated += EiiSorted[j]
#                 skip = j
#                 anyMatch = True
#
#         if not anyMatch:
#             skip = i
#
#         EiiReduced.append(EiiRrepeated)
#         Kreduced.append(KrSorted[i])
#         i = skip + 1
#
#     return EiiReduced, Kreduced
#
#
#
#
# # # Shift freqs all to non-negative
# # kX, kY = 2*np.pi*(freqX - freqX.min()), 2*np.pi*(freqY - freqY.min())
# #
# # krOld = 0
# # E, kr = np.zeros((uResFft.shape[0], 1)), np.zeros((uResFft.shape[0], 1))
# # for i in range(uResFft.shape[0]):
# #     kr[i] = np.sqrt(kX[i]**2 + kY[i]**2)
# #     dk = abs(krOld - kr[i])
# #     # This should depend on K
# #     eii = float(uResFft[i, i]*np.conj(uResFft[i, i])) + float(vResFft[i, i]*np.conj(vResFft[i, i]))
# #     E[i] = eii/2.
# #
# #     krOld = kr[i]
#
# plt.figure('uv')
# plt.loglog(Kreduced, EiiReduced)


# import numpy as np
# import vtk
# from vtk.numpy_interface import dataset_adapter as dsa
# from vtk.util import numpy_support as VN
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import meshio
#
# # mesh = meshio.read('./ABL_N_H/Slices/20060.9038025/U_slice_horizontal_2.vtk')
# reader = vtk.vtkPolyDataReader()
# reader.SetFileName('./ABL_N_H/Slices/20060.9038025/U_slice_horizontal_2.vtk')
# # reader.ReadAllVectorsOn()
# # reader.ReadAllScalarsOn()
# reader.Update()
#
# polydata = reader.GetOutput()
#
# cellArr = dsa.WrapDataObject(polydata).Polygons
#
# # ptLst = np.zeros((1, 3))
# # for i in range(polydata.GetNumberOfCells()):
# #    pts = polydata.GetCell(i).GetPoints()
# #    # cells = polydata.GetCell(i)
# #    np_pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
# #    ptLst = np.vstack((ptLst, np_pts))
# #    print(np_pts)
#
#
# # # dim = data.GetDimensions()
# # dim = (0, 0, 0)
# # vec = list(dim)
# # vec = [i-1 for i in dim]
# # vec.append(3)
# #
# # u = VN.vtk_to_numpy(polydata.GetCellData().GetArray('U'))
# # # uHor =
# # # b = VN.vtk_to_numpy(data.GetCellData().GetArray('POINTS'))
# #
# # # u = u.reshape((300, 300), order='F')
# # # b = b.reshape(vec,order='F')
# #
# # x = np.zeros(data.GetNumberOfPoints())
# # y = np.zeros(data.GetNumberOfPoints())
# # z = np.zeros(data.GetNumberOfPoints())
# #
# # # xMesh, yMesh = np.meshgrid(x, y)
# #
# # for i in range(data.GetNumberOfPoints()):
# #     x[i],y[i],z[i] = data.GetPoint(i)
# #
# # # Sort xy. First based on x, then y (x doesn't move)
# # xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
# #
# #
# # p2c = vtk.vtkPointDataToCellData()
# # p2c.SetInputConnection(reader.GetOutputPort())
# # p2c.Update()
# #
# # a = p2c.GetOutput()
# # b = VN.vtk_to_numpy(a.GetCellData().GetArray('POINTS'))
# # # iterate over blocks and copy in the result
# #
# # iter=dsa.MultiCompositeDataIterator([p2c.GetOutputDataObject(0), output])
# #
# # for  in_block,  output_block in iter:
# #
# #      output_block.GetCellData().AddArray(in_block.VTKObject.GetCellData().GetArray('DISPL'))
# #
# #
# #
# #
# # a = p2c
# # warp = vtk.vtkWarpVector()
# # b = warp.SetInputConnection(p2c.GetOutputPort())
# #
# # # for i, row in enumerate(xy):
# # #     if row[0] == 0:
# #
# #
# #
# # # plt.figure('x')
# # # plt.scatter(np.arange(0, x.shape[0]), x)
# # #
# # # plt.figure('y')
# # # plt.scatter(np.arange(0, x.shape[0]), y)
# #
# #
# # # x = x.reshape((301, 301), order='F')
# # # y = y.reshape((301, 301),order='F')
# # # z = z.reshape((301, 301),order='F')
# # #
# # # plt.figure()
# # # plt.contour(x[:-1, :-1], y[:-1, :-1], u)
