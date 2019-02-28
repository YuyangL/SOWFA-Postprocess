# from PostProcess_FieldData import FieldData
import numpy as np
import os
import matplotlib.pyplot as plt
import PostProcess_EnergySpectrum
import time
from Utilities import timer
from numba import jit, njit, prange
import pickle

"""
User Inputs
"""
caseDir, case = 'J:', 'ABL_N_L2'
sliceFolder, resultFolder = 'Slices', 'Result'
sliceName = 'U_hubHeight_Slice.raw'


"""
Process User Inputs
"""
caseFullPath = caseDir + '/' + case + '/' + sliceFolder + '/'
resultPath = caseFullPath + resultFolder + '/'


"""
Calculate Energy Spectrum
"""
@timer
@jit(parallel = True, fastmath = True)
def getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes, type = 'decomposed'):
    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()

    print(uRes2D.mean(), vRes2D.mean(), wRes2D.mean())

    #    # Normalized
    #    uRes2D, vRes2D, wRes2D = (u2D - u2D.mean())/u2D.mean(), (v2D - v2D.mean())/v2D.mean(), (w2D - w2D.mean(
    #    ))/w2D.mean()

    # Perform 2D FFT
    uResFft = np.fft.fft2(uRes2D, axes = (0, 1))
    vResFft = np.fft.fft2(vRes2D, axes = (0, 1))
    wResFft = np.fft.fft2(wRes2D, axes = (0, 1))
    # Shift FFT results so that 0-frequency component is in the center of the spectrum
    uResFft, vResFft, wResFft = np.fft.fftshift(uResFft), np.fft.fftshift(vResFft), np.fft.fftshift(wResFft)

    nX, nY = uRes2D.shape[1], uRes2D.shape[0]
    # Frequencies are number of rows/columns with a unit of cellSize meters
    # Note these are wavelength vector (Kx, Ky)
    freqX, freqY = np.fft.fftfreq(nX, d = cellSizes[0]), np.fft.fftfreq(nY, d = cellSizes[1])
    # Also shift corresponding frequencies
    freqX, freqY = np.fft.fftshift(freqX), np.fft.fftshift(freqY)

    # return uResFft, vResFft, wResFft, freqX, freqY




    # Calculate energy density Eii(Kx, Ky) (per d(Kx, Ky))
    # If 'decomposed' type, calculate E of horizontal velocity and vertical separately
    if type == 'decomposed':
        # Eii(Kx, Ky) = 0.5(|uResFft(Kx, Ky)^2| + ...)
        # u*np.conj(u) equals |u|^2
        # abs() to get the real part
        Eii = 0.5*np.abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft))
        #        # Not sure about this
        #        Eij = 0.5*np.abs(uResFft*np.conj(vResFft) + vResFft*np.conj(uResFft))
        # Vertical E separate from the horizontal one
        E33 = 0.5*np.abs(wResFft*np.conj(wResFft))
    else:
        Eii = 0.5*np.abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft) + wResFft*np.conj(wResFft))
        # Dummy array for vertical E33 and Eij
        Eij, E33 = np.empty((wResFft.shape[0], wResFft.shape[1])), np.empty((wResFft.shape[0], wResFft.shape[1]))

    # No Eij yet

    # Convert (Kx, Ky) to Kr for 1D energy spectrum result/plot
    # First, get all Kr[i_r] = sqrt(Kx^2 + Ky^2) and its corresponding Eii(Kr)
    Kr = np.empty(len(freqX)*len(freqY))
    Eii_r, E33_r = np.empty(len(freqX)*len(freqY)), np.empty(len(freqX)*len(freqY))
    i_r = 0
    for iX in prange(len(freqX)):
        for iY in range(len(freqY)):
            Kr[i_r] = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
            Eii_r[i_r], E33_r[i_r] = Eii[iY, iX], E33[iY, iX]
            i_r += 1

    # Then, sort Kr from low to high and sort Eii(Kr) accordingly
    KrSorted = np.sort(Kr)
    sortIdx = np.argsort(Kr)
    EiiSorted, E33Sorted = Eii_r[sortIdx], E33_r[sortIdx]

    # Lastly, find all Eii, Eij, E33 of equal Kr and line-integrate over co-centric Kr (would have been sphere for
    # slice not slice)
    E, Evert, KrFinal = [], [], []
    i = 0
    # Go through all Kr from low to high
    while i < len(KrSorted):
        # Remember E of current Kr[i]
        Eii_repeated, E33_repeated = EiiSorted[i], E33Sorted[i]
        match = 0.
        # Go through the rest of Kr other than Kr[i]
        for j in prange(i + 1, len(KrSorted)):
            # If the other Kr[j] matches current Kr[i]
            if KrSorted[j] == KrSorted[i]:
                Eii_repeated += EiiSorted[j]
                E33_repeated += E33Sorted[j]
                match += 1.

        # If multiple matches, i.e. if multiple E in the same cocentric ring, then get the average E
        if match > 0:
            Eii_repeated /= (match + 1)
            E33_repeated /= (match + 1)

        #        E.append(Eii_repeated*2*np.pi*KrSorted[i])
        #        Evert.append(E33_repeated*2*np.pi*KrSorted[i])
        E.append(Eii_repeated)
        Evert.append(E33_repeated)
        KrFinal.append(KrSorted[i])
        i += int(match) + 1

    print('\nEnergy spectrum calculated')
    return np.array(E), np.array(Evert), np.array(KrFinal)


@timer
# @jit(parallel = True, fastmath = True)
def getPlanarEnergySpectrum2(u2D, v2D, w2D, cellSizes, type = 'decomposed'):
    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()

    print(uRes2D.mean(), vRes2D.mean(), wRes2D.mean())

    #    # Normalized
    #    uRes2D, vRes2D, wRes2D = (u2D - u2D.mean())/u2D.mean(), (v2D - v2D.mean())/v2D.mean(), (w2D - w2D.mean(
    #    ))/w2D.mean()

    # # Calculate one-point (cross-)correlations
    # R11, R12, R13 = np.multiply(uRes2D, uRes2D), np.multiply(uRes2D, vRes2D), np.multiply(uRes2D, wRes2D)
    # R22, R23 = np.multiply(vRes2D, vRes2D), np.multiply(vRes2D, wRes2D)
    # R33 = np.multiply(wRes2D, wRes2D)
    # # The one-point correlation trace, equivalent to 2TKE(x, y)
    # Rii = R11 + R22 + R33

    # # Discrete Fourier transform of one-point (cross-)correlations
    # # The order in each direction is
    # R11Fft, R12Fft, R13Fft = np.fft.fft2(R11, axes = (0, 1)), np.fft.fft2(R12, axes = (0, 1)), np.fft.fft2(R13, axes = (0, 1))
    # R22Fft, R23Fft = np.fft.fft2(R22, axes = (0, 1)), np.fft.fft2(R23, axes = (0, 1))
    # R33Fft = np.fft.fft2(R33, axes = (0, 1))
    # RiiFft = np.fft.fft2(Rii, axes = (0, 1))

    U1Res2D = (uRes2D, uRes2D, uRes2D, vRes2D, vRes2D, wRes2D)
    U2Res2D = (uRes2D, vRes2D, wRes2D, vRes2D, wRes2D, wRes2D)
    nPtX, nPtY = uRes2D.shape[0], uRes2D.shape[0]
    # Take the lower bound integer of half of nPt, applies to both even and odd nPtX/nPtY
    nPtKx, nPtKy = nPtX//2, nPtY//2
    Rij = np.empty((nPtY, nPtX, 6))
    # DFT of Rij will be half of Rij sizes since the other half are just conjugates that's not interesting, given that Rij(x, y) are all real
    RijFft = np.empty((nPtKy, nPtKx, 6), dtype = np.complex128)
    # Rij = np.multiply(uRes2D, uRes2D)
    # RijFft = np.fft.fft2(np.multiply(uRes2D, uRes2D), axes = (0, 1))
    for i in prange(6):
        Rij[:, :, i] = np.multiply(U1Res2D[i], U2Res2D[i])
        # Take only the 1st half in each direction as the 2nd half is not interesting
        # RijFft[0, 0] is the sum of Rij by definition
        RijFft[:, :, i] = (np.fft.fft2(Rij[:, :, i], axes = (0, 1)))[:nPtKy, :nPtKx]

    Rii = Rij[:, :, 0] + Rij[:, :, 3] + Rij[:, :, 5]
    RiiFft = np.fft.fft2(Rii, axes = (0, 1))[:nPtKy, :nPtKx]
    # RiiFft[0, 0] is the sum of Rii by definition, which is 2TKE
    TKE = np.abs(0.5*RiiFft[0, 0])

    # Corresponding frequency in x and y directions, expressed in cycles/m
    # Number of columns are number of x
    # d is sample spacing, which should be equidistant,
    # in this case, cell size in x and y respectively
    # Again, only take the first half, the positive frequencies
    Kx, Ky = np.fft.fftfreq(nPtX, d = cellSizes[0])[:nPtKx], np.fft.fftfreq(nPtY, d = cellSizes[1])[:nPtKy]

    Kr, iKxy = np.empty(nPtKx*nPtKy), []
    cnt = 0
    for i in prange(len(Kx)):
        for j in range(len(Ky)):
            Kr[cnt] = np.sqrt(Kx[i]**2 + Ky[j]**2)
            # To access i/j, use iKxy[][] format
            iKxy.append((i, j))
            cnt += 1

    iKxy = np.array(iKxy)
    iKr_sorted = np.argsort(Kr, axis = 0)
    Kr_sorted, iKxy_sorted = Kr[iKr_sorted], iKxy[iKr_sorted]

    print('\nMerging x and y components into r...')
    # Sum of any RFft that has equal Kr
    # For each Kr_sorted[i]
    RijFft_r, RiiFft_r = [], []
    i = 0
    while i <= len(iKr_sorted) - 1:
        # Store current RFft value
        # iKxy_sorted[:, 1] is y
        RijFft_r_i, RiiFft_r_i = RijFft[iKxy_sorted[i, 1], iKxy_sorted[i, 0]], RiiFft[iKxy_sorted[i, 1], iKxy_sorted[i, 0]]
        match = 0
        # Go through every Kr_sorted[j, j > i]
        for j in range(i + 1, len(iKr_sorted)):
            # If Kr_sorted[j] = Kr_sorted[i], add up RFft
            if Kr_sorted[j] == Kr_sorted[i]:
                match += 1
                # Recall Kx/Ky is the same order of RFft's column/row
                RijFft_r_i += RijFft[iKxy_sorted[j, 1], iKxy_sorted[j, 0]]
                RiiFft_r_i += RiiFft[iKxy_sorted[j, 1], iKxy_sorted[j, 0]]
            # Since Kr_sorted is sorted from low to high, if there's no match, then Kr[i] is unique, proceed to Kr[i + 1]
            else:
                break

        RijFft_r.append(RijFft_r_i)
        RiiFft_r.append(RiiFft_r_i)
        i += match + 1

    # Get the unique Kr_sorted
    Kr_sorted = np.unique(Kr_sorted)
    Eij, Eii = np.array(RijFft_r), np.array(RiiFft_r)

    return RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr_sorted, TKE



t0 = time.time()
x2D, y2D, z2D, U2D, u2D, v2D, w2D = PostProcess_EnergySpectrum.readSliceRawData(sliceName = sliceName, case = case, caseDir = caseDir)

# E, Evert, Kr = PostProcess_EnergySpectrum.getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = np.array([10., 10.]))

t1 = time.time()
ticToc = t1 - t0

# uResFft, vResFft, wResFft, freqX, freqY = getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = [10., 10.])

RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr, TKE = getPlanarEnergySpectrum2(u2D, v2D, w2D, [10., 10.])

# Kr manipulation
Kr *= 10  # [CAUTION]

sampInterval, N = 10., 301
f = np.linspace(0, 1/(2*sampInterval), 301//2)

# Kolmogorov dimensional analysis: Lt ~ TKE^(3/2)/epsilon
# Lt is characteristic length scale of large eddies
# epsilon is dissipation rate
Lt = 300.
epsilon = Lt/(TKE**(3/2.))
# Kolmogorov -5/3 model
# E(Kr) = c_k*epsilon^(2/3)*Kr^(-5/3)
# c_k = 1.5 generally
E_Kolmo = 1.5*epsilon**(2/3.)*Kr**(-5/3)


plt.figure('Eii')
plt.ylabel("Eii")
plt.xlabel("Frequency 10Kr [cycle/m?]")
# plt.bar(f[:N//2], np.abs(uResFft)[:N//2]*1./N, width = 1.5)  # 1 / N is a normalization factor
# Kr manipulation here!
plt.loglog(Kr, np.abs(Eii)/len(Eii))
plt.loglog(Kr, E_Kolmo)
plt.xlim(1e-3, 1)
plt.ylim(1e-6, 1)
plt.show()

plt.figure('E12')
plt.ylabel("E12")
plt.xlabel("Frequency 10Kr [cycle/m?]")
# plt.bar(f[:N//2], np.abs(uResFft)[:N//2]*1./N, width = 1.5)  # 1 / N is a normalization factor
# E12 is second column of Eij
# Kr manipulation here!
plt.loglog(Kr, np.abs(Eij[:, 1])/len(Eij))
plt.loglog(Kr, E_Kolmo)
plt.xlim(1e-3, 1)
plt.ylim(1e-6, 1)
plt.show()

plt.figure('E33')
plt.ylabel("E33")
plt.xlabel("Frequency 10Kr [cycle/m?]")
# plt.bar(f[:N//2], np.abs(uResFft)[:N//2]*1./N, width = 1.5)  # 1 / N is a normalization factor
# E33 is last column of Eij
# Kr manipulation here!
plt.loglog(Kr, np.abs(Eij[:, -1])/len(Eij))
plt.loglog(Kr, E_Kolmo)
plt.xlim(1e-3, 1)
plt.ylim(1e-6, 1)
plt.show()

# E, Evert, Kr = getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = [10., 10.])
#
# pickle.dump(E, open(resultPath + 'E.p', 'wb'))
# pickle.dump(Evert, open(resultPath + 'Evert.p', 'wb'))
# pickle.dump(Kr, open(resultPath + 'Kr.p', 'wb'))
#
# """
# Plot Energy Spectrum
# """
# plt.figure('horizontal')
# plt.loglog(Kr, E)
# # plt.loglog(Kr, Kr**(-5/3.)*(9e-3)**(2/3.)*1.5)
# plt.xlim(1e-3, 1)
# plt.tight_layout()
#
# plt.figure('vertical')
# plt.loglog(Kr, Evert)
# # plt.loglog(Kr, Kr**(-5/3.)*(9e-3)**(2/3.)*1.5)
# plt.xlim(1e-3, 1)
# plt.tight_layout()


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
