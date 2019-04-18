import os
# cimport numpy imports Numpy's C API but doesn't incl. all Numpy functions
import numpy as np
cimport numpy as np
#from cpython cimport bool
cimport cython
from cython.parallel cimport prange
#from libc.math cimport sqrt, abs, mean
from libc.math cimport sqrt

#DTYPE = np.int
#ctypedef np.int_t DTYPE_t

# cpdef is a hybrid Python-C function
# Only use cdef (fastest) for functions if you don't intend to call it outside Cython
# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
cpdef tuple readStructuredSliceData(str sliceName, str case = 'ABL_N_H', str caseDir = '.', str time = 'auto', str resultFolder = 'Result', str sliceFolder = 'Slices'):
    cdef str sliceFullPath
    cdef np.ndarray[np.float_t] row, scalarField
    # The following need numpy reshape method, thus not memoryview
    cdef np.ndarray[np.float_t] x, y, z, u, v, w
    cdef np.ndarray[np.float_t, ndim = 2] data, x2D, y2D, z2D, u2D, v2D, w2D, scalarField2D
    cdef double valOld, val
    cdef int i, nPtX, nPtY
    cdef str caseFullPath = caseDir + '/' + case + '/' + sliceFolder + '/'
    cdef str resultPath = caseFullPath + resultFolder + '/'

    # Try making the result folder, if it doesn't exist
    try:
        os.makedirs(resultPath)
    except OSError:
        pass

    # If time is 'auto', pick the 1st from the available times
    time = os.listdir(caseFullPath)[0] if time is 'auto' else time
    # Full path to the slice
    sliceFullPath = caseFullPath + time + '/' + sliceName
    # Read slice data, headers with # are auto trimmed
    data = np.genfromtxt(sliceFullPath)
    # 1D array
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    # Mesh size in x
    # Since the slice is sorted from low to high x, count the number of x
    valOld = x[0]
    for i, val in enumerate(x[1:]):
        if val < valOld:
            nPtX = i + 1
            break

        valOld = val

    nPtY = x.shape[0]/nPtX
    x2D, y2D, z2D = x.reshape((nPtY, nPtX)), y.reshape((nPtY, nPtX)), z.reshape((nPtY, nPtX))
#    if data.shape[1] == 6:
    u, v, w = data[:, 3], data[:, 4], data[:, 5]
    scalarField = np.empty(data.shape[0])
    # Go through every row and calculate resultant value
    # nogil doesn't support numpy
    # Using sqrt from clib.math instead, for 1D array
    for i in prange(data.shape[0], nogil = True):
        scalarField[i] = sqrt(data[i, 3]**2 + data[i, 4]**2 + data[i, 5]**2)
#    for i, row in enumerate(data):
#        scalarField[i] = np.sqrt(data[i][3]**2 + row[4]**2 + row[5]**2)
#    else:
#        u, v, w = np.zeros((data.shape[1], 1), dtype = np.double), np.zeros((data.shape[1], 1), dtype = np.double), np.zeros((data.shape[1], 1), dtype = np.double)
#        scalarField = data[:, 3]

    u2D, v2D, w2D = u.reshape((nPtY, nPtX)), v.reshape((nPtY, nPtX)), w.reshape((nPtY, nPtX))
    scalarField2D = scalarField.reshape((nPtY, nPtX))

    print('\nSlice raw data read')
    return x2D, y2D, z2D, scalarField2D, u2D, v2D, w2D


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef getPlanarEnergySpectrum(np.ndarray[np.float_t, ndim= 2] u2D, np.ndarray[np.float_t, ndim= 2] v2D, np.ndarray[np.float_t, ndim= 2] w2D, double L, tuple cellSizes2D, horizontalEii = False):
    cdef np.ndarray[np.float_t, ndim = 2] uRes2D, vRes2D, wRes2D, KrOld
    cdef np.ndarray[np.complex128_t, ndim = 2] uResFft, vResFft, wResFft
    cdef double TKE, Kr0
    cdef int nPtX, nPtY, i, j
    cdef tuple U1ResFft, U2ResFft
    cdef np.ndarray[np.complex128_t, ndim = 2] RiiFft, Eij
    cdef np.ndarray[np.complex128_t, ndim = 3] RijFft
    cdef np.ndarray[np.complex128_t] Eii
    cdef np.ndarray[np.float_t] Kr, occurrence

    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
    # TKE calculated form physical space
    TKE = 0.5*np.sum(uRes2D**2 + vRes2D**2 + wRes2D**2)
    # Number of samples in x and y/z, columns correspond to x
    nPtX, nPtY = uRes2D.shape[1], uRes2D.shape[0]
    # 2D DFT, no normalization (will be done manually below)
    uResFft, vResFft, wResFft = np.fft.fft2(uRes2D, axes = (0, 1), norm = None), \
                                np.fft.fft2(vRes2D, axes = (0, 1), norm = None), \
                                np.fft.fft2(wRes2D, axes = (0, 1), norm = None)
    # Normalization by N^(dimension)
    uResFft /= (nPtX*nPtY)
    vResFft /= (nPtX*nPtY)
    wResFft /= (nPtX*nPtY)

    # Corresponding frequency in x and y/z directions, expressed in cycles/m
    # Kx corresponds to k/n in np.fft.fft documentation, where
    # k = 0, ..., (n - 1)/2; n is number of samples
    # Number of columns are number of x
    # d is sample spacing, which should be equidistant,
    # in this case, cell size in x and y/z respectively
    Kx, Ky = np.fft.fftfreq(nPtX, d = cellSizes2D[0]), np.fft.fftfreq(nPtY, d = cellSizes2D[1])
    # Kx and Ky/Kz is defined as 2n*pi/L, while the K in np.fft.fftn() is simply n/L, n in [1, N]
    # Thus scale old Kx, Ky/Kz by 2pi and 2D meshgrid treatment
    Kx *= 2*np.pi
    Ky *= 2*np.pi
    Kx, Ky = np.meshgrid(Kx, Ky)
    # Before calculating (cross-)correlation, add arrays to 2 tuples
    U1ResFft = (uResFft, uResFft, uResFft, vResFft, vResFft, wResFft)
    U2ResFft = (uResFft, vResFft, wResFft, vResFft, wResFft, wResFft)
    # Initialize 2D Rij in spectral space
    RijFft = np.empty((nPtY, nPtX, 6), dtype = np.complex128)
    # Go through each component of RijFft
    # The 6 components are 11, 12, 13,
    # 22, 23,
    # 33
    for i in range(6):
        # Perform the 2-point (cross-)correlation
        RijFft[:, :, i] = np.multiply(U1ResFft[i], np.conj(U2ResFft[i]))

    # Trace of 2-point correlations
    # If decompose Rii to horizontal Rii and R33
    if horizontalEii:
        RiiFft = RijFft[:, :, 0] + RijFft[:, :, 3]
    else:
        RiiFft = RijFft[:, :, 0] + RijFft[:, :, 3] + RijFft[:, :, 5]

    # Original resultant Kr
    KrOld = np.sqrt(Kx**2 + Ky**2)
    # New proposed Kr for E spectrum, same number of points as x
    Kr0 = 2*np.pi/L
    Kr = Kr0*np.linspace(1, nPtX, nPtX)
    # Initialize Eij combined from 2D to 1D
    # Eij[i] is 0.5ui'uj' = 0.5sum(Rij of equal Kr[i])
    Eij = np.empty((len(Kr), 6), dtype = np.complex128)
    Eii = np.empty_like(Kr, dtype = np.complex128)
    # Occurrence when KrOld is close to each Kr[i]
    occurrence = np.empty(len(Kr))
    # Go through each proposed Kr
    # Integrate Rij where KrOld lies between Kr0*[(i + 1) - 0.5, (i + 1) + 0.5)
    # This is possible since RijFft and KrOld has matching 2D indices
    for i in range(len(Kr)):
        occurrence[i] = len(KrOld[(KrOld >= Kr0*(i + 1 - 0.5)) & (KrOld < Kr0*(i + 1 + 0.5))])
        # For Eij, go through all 6 components
        for j in range(6):
            Eij[i, j] = 0.5*np.sum(RijFft[:, :, j][(KrOld >= Kr0*(i + 1 - 0.5)) & (KrOld < Kr0*(i + 1 + 0.5))])

        Eii[i] = 0.5*np.sum(RiiFft[(KrOld >= Kr0*(i + 1 - 0.5)) & (KrOld < Kr0*(i + 1 + 0.5))])

    # # If done right, TKE from energy spectrum should equal TKE in physical space
    # TKE_k = sum(Eii)*nPtX*nPtY

    # # Optional equal Kr histogram
    # plt.figure('Histogram')
    # plt.hist(occurrence)

    return RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr, TKE
