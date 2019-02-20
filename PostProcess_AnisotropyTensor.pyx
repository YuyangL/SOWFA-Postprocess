import numpy as np
cimport numpy as np
cimport cython

# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
cpdef tuple processAnisotropyTensor(np.ndarray[np.float_t, ndim = 3] vals3D):
    # If ndim is not provided but np.float_t is provided, 1D is assumed
    cdef np.ndarray[np.float_t, ndim = 2] k, eigVecs
    cdef np.ndarray[np.float_t, ndim = 1] eigVals
    cdef np.ndarray tensors, eigValsGrid, eigVecsGrid
    cdef int i, j

    # TKE in the interpolated mesh
    # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
    k = 0.5*(vals3D[:, :, 0] + vals3D[:, :, 3] + vals3D[:, :, 5])
    # Convert Rij to bij
    for i in range(6):
        vals3D[:, :, i] = vals3D[:, :, i]/(2.*k) - 1/3. if i in (0, 3, 5) else vals3D[:, :, i]/(2.*k)

    # Add each anisotropy tensor to each mesh grid location, in depth
    # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
    tensors = np.dstack((vals3D[:, :, 0], vals3D[:, :, 1], vals3D[:, :, 2],
                         vals3D[:, :, 1], vals3D[:, :, 3], vals3D[:, :, 4],
                         vals3D[:, :, 2], vals3D[:, :, 4], vals3D[:, :, 5]))
    # Reshape the z dir to 3x3 instead of 9x1
    # Now tensors is 4D, with x, y being mesh grid, z1, z2 being the 3x3 tensor at (x, y)
    tensors = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))

    # Evaluate eigenvalues and eigenvectors of the symmetric tensor
    # eigVecsGrid is nX x nY x 9, where 9 is the flattened eigenvector matrix from np.linalg.eigh()
    eigValsGrid, eigVecsGrid = np.zeros(3), np.zeros((tensors.shape[0], tensors.shape[1], 9))
    for i in range(tensors.shape[0]):
        for j in range(tensors.shape[1]):
            # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
            # Each col of eigVecs is a vector, thus 3 x 3
            eigVals, eigVecs = np.linalg.eigh(tensors[i, j, :, :])
            eigVals, eigVecs = np.flipud(eigVals), np.fliplr(eigVecs)
            # Each eigVals is a row, stack them vertically
            # Each eigVecs is a 3 x 3 matrix, stack them in z dir to each of their i, j location
            eigValsGrid = np.vstack((eigValsGrid, eigVals))
            eigVecsGrid[i, j, :] = eigVecs.ravel()

    # Reshape eigValsGrid to nRow x nCol x 3
    # so that each mesh grid location has 3 eigenvalues
    # Remove the first row since it was dummy
    # Also reshape eigVecsGrid from nRow x nCol x 9 to nRow x nCol x 3 x 3
    # so that each col of the 3 x 3 matrix is an eigenvector corresponding to an eigenvalue
    eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))
    eigVecsGrid = np.reshape(eigVecsGrid, (eigVecsGrid.shape[0], eigVecsGrid.shape[1], 3, 3))

    print('\nFinished processing anisotropy tensors')
    return vals3D, tensors, eigValsGrid, eigVecsGrid


# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
cpdef tuple processAnisotropyTensor_Uninterpolated(np.ndarray[np.float_t, ndim = 2] vals2D, int realizeIter = 0):
    # If ndim is not provided but np.float_t is provided, 1D is assumed
    cdef np.ndarray[np.float_t] k, eigVals
    cdef np.ndarray[np.float_t, ndim = 2] eigVecs
    cdef np.ndarray tensors, eigVals3D, eigVecs4D
    cdef int i

    # TKE
    # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
    k = 0.5*(vals2D[:, 0] + vals2D[:, 3] + vals2D[:, 5])
    # Convert Rij to bij
    for i in range(6):
        vals2D[:, i] = vals2D[:, i]/(2.*k) - 1/3. if i in (0, 3, 5) else vals2D[:, i]/(2.*k)

    # Add each anisotropy tensor to each mesh grid location, in depth
    # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
    tensors = np.dstack((vals2D[:, 0], vals2D[:, 1], vals2D[:, 2],
                         vals2D[:, 1], vals2D[:, 3], vals2D[:, 4],
                         vals2D[:, 2], vals2D[:, 4], vals2D[:, 5]))

    tensors = tensors.reshape((tensors.shape[1], 9))
    for i in range(realizeIter):
        print('\nApplying realizability filter ' + str(i + 1))
        tensors = make_realizable(tensors)

    # Reshape the z dir to 3x3 instead of 9x1
    # Now tensors is 4D, with x, y being nRow, 1, z1, z2 being the 3x3 tensor at (x, y)
    tensors = tensors.reshape((tensors.shape[0], 1, 3, 3))

    # Evaluate eigenvalues and eigenvectors of the symmetric tensor
    # eigVecs4D is nX x nY x 9, where 9 is the flattened eigenvector matrix from np.linalg.eigh()
    eigVals3D, eigVecs4D = np.zeros(3), np.zeros((tensors.shape[0], 1, 9))
    for i in range(tensors.shape[0]):
        # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
        # Each col of eigVecs is a vector, thus 3 x 3
        eigVals, eigVecs = np.linalg.eigh(tensors[i, 0, :, :])
        eigVals, eigVecs = np.flipud(eigVals), np.fliplr(eigVecs)
        # Each eigVals is a row, stack them vertically
        # Each eigVecs is a 3 x 3 matrix, stack them in z dir to each of their i, j = 0 location
        eigVals3D = np.vstack((eigVals3D, eigVals))
        eigVecs4D[i, 0, :] = eigVecs.ravel()

    # Reshape eigVals3D to nRow x 1 x 3
    # so that each mesh grid location has 3 eigenvalues
    # Remove the first row since it was dummy
    # Also reshape eigVecs4D from nRow x 1 x 9 to nRow x 1 x 3 x 3
    # so that each col of the 3 x 3 matrix is an eigenvector corresponding to an eigenvalue
    eigVals3D = np.reshape(eigVals3D[1:], (tensors.shape[0], 1, 3))
    eigVecs4D = np.reshape(eigVecs4D, (tensors.shape[0], 1, 3, 3))

    print('\nFinished processing anisotropy tensors')
    return vals2D, tensors, eigVals3D, eigVecs4D


# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
cdef np.ndarray[np.float_t, ndim = 2] make_realizable(np.ndarray[np.float_t, ndim = 2] labels):
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn:
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.
    :param labels: the predicted anisotropy tensor (num_points X 9 array)
    """
    cdef int numPoints, i, j
    cdef np.ndarray[np.float_t, ndim = 2] A, evectors
    cdef np.ndarray[np.float_t] evalues

    numPoints = labels.shape[0]
    A = np.zeros((3, 3))
    for i in range(numPoints):
        # Scales all on-diags to retain zero trace
        if np.min(labels[i, [0, 4, 8]]) < -1./3.:
            labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
        if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
            labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
        if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
            labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
        if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
            labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
            labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = labels[i, 0]
        A[1, 1] = labels[i, 4]
        A[2, 2] = labels[i, 8]
        A[0, 1] = labels[i, 1]
        A[1, 0] = labels[i, 1]
        A[1, 2] = labels[i, 5]
        A[2, 1] = labels[i, 5]
        A[0, 2] = labels[i, 2]
        A[2, 0] = labels[i, 2]
        evalues, evectors = np.linalg.eig(A)
        if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/2.:
            evalues = evalues*(3.*np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])/(2.*np.max(evalues))
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]
        if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
            evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]

    return labels


