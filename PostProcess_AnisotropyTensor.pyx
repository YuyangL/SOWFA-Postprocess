import numpy as np
cimport numpy as np
cimport cython

# Don't check for array bounds
@cython.boundscheck(False)
cpdef tuple processAnisotropyTensor(dict valsDecomp):
    # If ndim is not provided, 1D is assumed
    cdef np.ndarray[np.float_t, ndim = 2] k, val
    cdef np.ndarray[np.float_t, ndim = 1] eigVals
    cdef np.ndarray tensors, eigValsGrid
    cdef str key
    cdef int i, j

    # TKE in the interpolated mesh
    # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
    k = 0.5*(valsDecomp['0'] + valsDecomp['3'] + valsDecomp['5'])
    # Convert Rij to bij
    for key, val in valsDecomp.items():
        valsDecomp[key] = val/(2.*k) - 1/3. if key in ('0', '3', '5') else val/(2.*k)

    # Add each anisotropy tensor to each mesh grid location, in depth
    # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
    tensors = np.dstack((valsDecomp['0'], valsDecomp['1'], valsDecomp['2'],
                      valsDecomp['1'], valsDecomp['3'], valsDecomp['4'],
                      valsDecomp['2'], valsDecomp['4'], valsDecomp['5']))
    # Reshape the z dir to 3x3 instead of 9x1
    # Now tensors is 4D, with x, y being mesh grid, z1, z2 being the 3x3 tensor at (x, y)
    tensors = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))

    # Evaluate eigenvalues of symmetric tensor
    eigValsGrid = np.zeros(3)
    for i in range(tensors.shape[0]):
        for j in range(tensors.shape[1]):
            # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
            eigVals = np.flipud(np.linalg.eigvalsh(tensors[i, j, :, :]))
            # Each eigVals is a row, stack them vertically
            eigValsGrid = np.vstack((eigValsGrid, eigVals))

    # Reshape eigVals to nRow x nCol x 3
    # so that each mesh grid location has 3 eigenvalues
    # Remove the first row since it was dummy
    eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))

    return valsDecomp, tensors, eigValsGrid
