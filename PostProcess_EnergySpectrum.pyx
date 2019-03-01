import os
# cimport numpy imports Numpy's C API but doesn't incl. all Numpy functions
import numpy as np
cimport numpy as np
#from cpython cimport bool
cimport cython
#from libc.math cimport sqrt, abs, mean

#DTYPE = np.int
#ctypedef np.int_t DTYPE_t

# cpdef is a hybrid Python-C function
# Only use cdef (fastest) for functions if you don't intend to call it outside Cython
# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
cpdef tuple readSliceRawData(str sliceName, str case = 'ABL_N_H', str caseDir = '.', str time = 'auto', str resultFolder = 'Result'):
    cdef str sliceFullPath
    cdef np.ndarray[np.float_t] row, scalarField
    # The following need numpy reshape method, thus not memoryview
    cdef np.ndarray[np.float_t] x, y, z, u, v, w
    cdef np.ndarray[np.float_t, ndim = 2] data, x2D, y2D, z2D, u2D, v2D, w2D, scalarField2D
    cdef double valOld, val
    cdef int i, nPtX, nPtY
    cdef str caseFullPath = caseDir + '/' + case + '/' + 'Slices/'
    cdef str resultPath = caseFullPath + resultFolder + '/'

    # Try making the result folder, if it doesn't exist
    try:
        os.makedirs(resultFolder)
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
    for i, row in enumerate(data):
        scalarField[i] = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2)
#    else:
#        u, v, w = np.zeros((data.shape[1], 1), dtype = np.double), np.zeros((data.shape[1], 1), dtype = np.double), np.zeros((data.shape[1], 1), dtype = np.double)
#        scalarField = data[:, 3]

    u2D, v2D, w2D = u.reshape((nPtY, nPtX)), v.reshape((nPtY, nPtX)), w.reshape((nPtY, nPtX))
    scalarField2D = scalarField.reshape((nPtY, nPtX))

    print('\nSlice raw data read')
    return x2D, y2D, z2D, scalarField2D, u2D, v2D, w2D

# [DEPRECATED] Refer to Visual_EnergySpectrum.getPlanarEnergySpectrum()
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple getSliceEnergySpectrum(np.ndarray[np.float_t, ndim = 2] u2D, np.ndarray[np.float_t, ndim = 2] v2D, np.ndarray[np.float_t, ndim = 2] w2D, list cellSizes, str type = 'decomposed'):
    cdef np.ndarray[np.float_t, ndim = 2] uRes2D, vRes2D, wRes2D
    cdef np.ndarray[np.float_t, ndim = 2] uResFft, vResFft, wResFft
    cdef np.ndarray[np.float_t] freqX, freqY
    cdef np.ndarray[np.float_t] Kr, Eii_r, Eij_r, E33_r, KrSorted, EiiSorted, EijSorted, E33Sorted
    cdef np.ndarray[np.int_t] sortIdx
    cdef np.ndarray[np.float_t, ndim = 2] Eii, Eij, E33
    cdef int nX, nY, i_r, i, j, iNew
    cdef double Eii_repeated, Eij_repeated, E33_repeated, match

    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
#    # Normalized
#    uRes2D, vRes2D, wRes2D = (u2D - u2D.mean())/u2D.mean(), (v2D - v2D.mean())/v2D.mean(), (w2D - w2D.mean())/w2D.mean()
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
        Eij, E33 = np.zeros((wResFft.shape[0], wResFft.shape[1])), np.zeros((wResFft.shape[0], wResFft.shape[1]))

    # No Eij yet
    # Convert (Kx, Ky) to Kr for 1D energy spectrum result/plot
    # First, get all Kr[i_r] = sqrt(Kx^2 + Ky^2) and its corresponding Eii(Kr)
    Kr = np.empty(len(freqX)*len(freqY))
    Eii_r, E33_r = np.empty(len(freqX)*len(freqY)), np.empty(len(freqX)*len(freqY))
    i_r = 0
    for iX in range(len(freqX)):
        for iY in range(len(freqY)):
            Kr[i_r] = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
            Eii_r[i_r], E33_r[i_r] = Eii[iY, iX], E33[iY, iX]
            i_r += 1

    # Then, sort Kr from low to high and sort Eii(Kr) accordingly
    KrSorted = np.sort(Kr)
    sortIdx = np.argsort(Kr)
    EiiSorted, E33Sorted = Eii_r[sortIdx], E33_r[sortIdx]
    # Lastly, find all Eii, Eij, E33 of equal Kr and line-integrate over co-centric Kr (would have been sphere for slice not slice)
    E, Evert, KrFinal = [], [], []
    i = 0
    # Go through all Kr from low to high
    while i < len(KrSorted):
        # Remember E of current Kr[i]
        Eii_repeated, E33_repeated = EiiSorted[i], E33Sorted[i]
        match = 0.
        # Go through the rest of Kr other than Kr[i]
        for j in range(i + 1, len(KrSorted)):
            # If the other Kr[j] matches current Kr[i]
            if KrSorted[j] == KrSorted[i]:
                Eii_repeated += EiiSorted[j]
                E33_repeated += E33Sorted[j]
                iNew = j
                match += 1.

        # If Kr[i] is unique, no change to new i
        if match == 0:
            iNew = i
        # If multiple matches, i.e. if multiple E in the same cocentric ring, then get the average E
        else:
            Eii_repeated /= (match + 1)
            E33_repeated /= (match + 1)

#        E.append(Eii_repeated*2*np.pi*KrSorted[i])
#        Evert.append(E33_repeated*2*np.pi*KrSorted[i])
        E.append(Eii_repeated)
        Evert.append(E33_repeated)
        KrFinal.append(KrSorted[i])
        i = iNew + 1

    print('\nEnergy spectrum calculated')
    return np.array(E), np.array(Evert), np.array(KrFinal)
