import os
# cimport numpy imports Numpy's C API but doesn't incl. all Numpy functions
import numpy as np
cimport numpy as np
#from cpython cimport bool
cimport cython

#DTYPE = np.int
#ctypedef np.int_t DTYPE_t

# cpdef is a hybrid Python-C function
# Only use cdef (fastest) for functions if you don't intend to call it outside Cython
# Don't check for array bounds
@cython.boundscheck(False)
cpdef tuple readSliceRawData(str field, str case = 'ABL_N_H/Slices', str caseDir = './', time = None, int skipHeader = 2):
    cdef str timePath, fieldFullPath
    # If ndim is not provided, 1D is assumed
    cdef np.ndarray[np.float_t] x, y, z
    cdef np.ndarray u, v, w,
    cdef np.ndarray[np.float_t, ndim = 2] data, scalarField, x2D, y2D, z2D, u2D, v2D, w2D, scalarField2D
    cdef double valOld, val
    cdef int i, nPtX

    timePath = caseDir + '/' + case + '/'
    if time is None:
        time = os.listdir(timePath)
        try:
            time.remove('Result')
        except:
            pass

        time = time[0]

    fieldFullPath = timePath + str(time) + '/' + field
    data = np.genfromtxt(fieldFullPath, skip_header = skipHeader, dtype = np.double)

    # 1D array
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # Mesh size in x
    valOld = x[0]
    for i, val in enumerate(x[1:]):
        if val < valOld:
            nPtX = i + 1
            break

        valOld = val

    x2D, y2D, z2D = x.reshape((-1, nPtX)), y.reshape((-1, nPtX)), z.reshape((-1, nPtX))

    if data.shape[1] == 6:
        u, v, w = data[:, 3], data[:, 4], data[:, 5]
        scalarField = np.zeros((data.shape[0], 1))
        for i, row in enumerate(data):
            scalarField[i] = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2)

    else:
        u, v, w = np.zeros((data.shape[1], 1), dtype = np.double), np.zeros((data.shape[1], 1), dtype = np.double), np.zeros((data.shape[1], 1), dtype = np.double)
        scalarField = data[:, 3]

    u2D, v2D, w2D = u.reshape((-1, nPtX)), v.reshape((-1, nPtX)), w.reshape((-1, nPtX))
    scalarField2D = scalarField.reshape((-1, nPtX))

    print('\nSlice raw data read')
    return x2D, y2D, z2D, scalarField2D, u2D, v2D, w2D


cpdef tuple getSliceEnergySpectrum(np.ndarray[np.float_t, ndim = 2] u2D, np.ndarray[np.float_t, ndim = 2] v2D, np.ndarray[np.float_t, ndim = 2] w2D, tuple cellSizes, str type = 'decomposed'):
    cdef np.ndarray[np.float_t, ndim = 2] uRes2D, vRes2D, wRes2D
    cdef np.ndarray[np.float_t, ndim = 2] uResFft, vResFft, wResFft
    cdef np.ndarray[np.float_t] freqX, freqY
    cdef np.ndarray[np.float_t, ndim = 2] Kr, KrSorted
    cdef np.ndarray[np.int] sortIdx
    cdef np.ndarray[np.float_t, ndim = 2] Eii, Eii_r, EiiSorted, E33, E33_r, E33Sorted
    cdef int nX, nY, i_r, i, j, skip
    cdef double Eii_repeated, E33_repeated, match
#    cdef bool anyMatch

    # Velocity fluctuations
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
#    # Normalized
#    uRes2D, vRes2D, wRes2D = (u2D - u2D.mean())/u2D.mean(), (v2D - v2D.mean())/v2D.mean(), (w2D - w2D.mean())/w2D.mean()


    # Perform 2D FFT
    uResFft = np.fft.fft2(uRes2D)
    vResFft = np.fft.fft2(vRes2D)
    wResFft = np.fft.fft2(wRes2D)
    # Shift FFT results
    uResFft, vResFft, wResFft = np.fft.fftshift(uResFft), np.fft.fftshift(vResFft), np.fft.fftshift(wResFft)

    nX, nY = uRes2D.shape[1], uRes2D.shape[0]
    # Frequencies are number of rows/columns with a unit of cellSize meters
    # Note these are wavelength vector (Kx, Ky)
    freqX, freqY = np.fft.fftfreq(nX, d = cellSizes[0]), np.fft.fftfreq(nY, d = cellSizes[1])
    # Also shift corresponding frequencies
    freqX, freqY = np.fft.fftshift(freqX), np.fft.fftshift(freqY)

    # Calculate energy density Eii(Kx, Ky) (per d(Kx, Ky))
    # If 'decomposed' type, calculate E of horizontal velocity and vertical separately
    if type is 'decomposed':
        # Eii(Kx, Ky) = 0.5(|uResFft(Kx, Ky)^2| + ...)
        # u*conj(u) equals |u|^2
        # abs() to get the real part
        Eii = 0.5*abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft))
        # Vertical E separate from the horizontal one
        E33 = 0.5*abs(wResFft*np.conj(wResFft))
    else:
        Eii = 0.5*abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft) + wResFft*np.conj(wResFft))
        # Dummy array for vertical E
        E33 = np.zeros((wResFft.shape[0], wResFft.shape[1]))

    # Convert (Kx, Ky) to Kr for 1D energy spectrum result/plot
    # First, get all Kr[i_r] = sqrt(Kx^2 + Ky^2) and its corresponding Eii(Kr)
    Kr = np.zeros((len(freqX)*len(freqY), 1))
    Eii_r, E33_r = np.zeros((len(freqX)*len(freqY), 1)), np.zeros((len(freqX)*len(freqY), 1))
    i_r = 0
    for iX in range(len(freqX)):
        for iY in range(len(freqY)):
            Kr[i_r] = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
            Eii_r[i_r], E33_r[i_r] = Eii[iY, iX], E33[iY, iX]
            i_r += 1

    # Then, sort Kr from low to high and sort Eii(Kr) accordingly
    KrSorted = np.sort(Kr, axis = 0)
    sortIdx = np.argsort(Kr, axis = 0).ravel()
    EiiSorted, E33Sorted = Eii_r[sortIdx], E33_r[sortIdx]

    # Lastly, find all Eii of equal Kr and line-integrate over co-centric Kr (would have been sphere for field not slice)
    E, Evert, KrFinal = [], [], []
    i = 0
    while i < len(KrSorted):
        Eii_repeated, E33_repeated = EiiSorted[i], E33Sorted[i]
        match = 0.
        for j in range(i + 1, len(KrSorted)):
            if KrSorted[j] == KrSorted[i]:
                Eii_repeated += EiiSorted[j]
                E33_repeated += E33Sorted[j]
                skip = j
                match += 1.

        if match == 0:
            skip = i
        else:
            Eii_repeated /= (match + 1)
            E33_repeated /= (match + 1)

        E.append(Eii_repeated*2*np.pi*KrSorted[i])
        Evert.append(E33_repeated*2*np.pi*KrSorted[i])
        KrFinal.append(KrSorted[i])
        i = skip + 1

    print('\nEnergy spectrum calculated')
    return np.array(E), np.array(Evert), np.array(KrFinal)
