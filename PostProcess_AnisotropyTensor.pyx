# cython: language_level = 3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from libc.stdio cimport printf
from libc.math cimport sqrt

# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple processAnisotropyTensor(np.ndarray[np.float_t, ndim = 3] vals3D):
    """
    [DEPRECATED]
    :param vals3D: 
    :type vals3D: 
    :return: 
    :rtype: 
    """
    # If ndim is not provided but np.float_t is provided, 1D is assumed
    cdef np.ndarray[np.float_t, ndim = 2] k, eigvec
    cdef np.ndarray[np.float_t, ndim = 1] eigVals
    cdef np.ndarray bij, eigValsGrid, eigvecGrid
    cdef int i, j, milestone
    cdef double progress

    print('\nProcessing anisotropy bij...')
    # TKE in the interpolated mesh
    # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
    k = 0.5*(vals3D[:, :, 0] + vals3D[:, :, 3] + vals3D[:, :, 5])
    # Avoid FPE
    k[k < 1e-8] = 1e-8
    # Convert Rij to bij
    for i in range(6):
        vals3D[:, :, i] = vals3D[:, :, i]/(2.*k) - 1/3. if i in (0, 3, 5) else vals3D[:, :, i]/(2.*k)

    # Add each anisotropy tensor to each mesh grid location, in depth
    # bij is 3D with z being b11, b12, b13, b21, b22, b23...
    bij = np.dstack((vals3D[:, :, 0], vals3D[:, :, 1], vals3D[:, :, 2],
                         vals3D[:, :, 1], vals3D[:, :, 3], vals3D[:, :, 4],
                         vals3D[:, :, 2], vals3D[:, :, 4], vals3D[:, :, 5]))
    # Reshape the z dir to 3x3 instead of 9x1
    # Now bij is 4D, with x, y being mesh grid, z1, z2 being the 3x3 tensor at (x, y)
    bij = bij.reshape((bij.shape[0], bij.shape[1], 3, 3))

    # Evaluate eigenvalues and eigenvectors of the symmetric tensor
    # eigvecGrid is nX x nY x 9, where 9 is the flattened eigenvector matrix from np.linalg.eigh()
    eigValsGrid, eigvecGrid = np.zeros(3), np.zeros((bij.shape[0], bij.shape[1], 9))
    # For gauging progress
    milestone = 10
    for i in range(bij.shape[0]):
        for j in range(bij.shape[1]):
            # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
            # Each col of eigvec is a vector, thus 3 x 3
            eigVals, eigvec = np.linalg.eigh(bij[i, j, :, :])
            eigVals, eigvec = np.flipud(eigVals), np.fliplr(eigvec)
            # Each eigVals is a row, stack them vertically
            # Each eigvec is a 3 x 3 matrix, stack them in z dir to each of their i, j location
            eigValsGrid = np.vstack((eigValsGrid, eigVals))
            eigvecGrid[i, j, :] = eigvec.ravel()

        # Gauge progress
        progress = float(i)/(bij.shape[0] + 1)*100.
        if progress >= milestone:
            print(' ' + str(milestone) + '%...')
            milestone += 10

    # Reshape eigValsGrid to nRow x nCol x 3
    # so that each mesh grid location has 3 eigenvalues
    # Remove the first row since it was dummy
    # Also reshape eigvecGrid from nRow x nCol x 9 to nRow x nCol x 3 x 3
    # so that each col of the 3 x 3 matrix is an eigenvector corresponding to an eigenvalue
    eigValsGrid = np.reshape(eigValsGrid[1:], (bij.shape[0], bij.shape[1], 3))
    eigvecGrid = np.reshape(eigvecGrid, (eigvecGrid.shape[0], eigvecGrid.shape[1], 3, 3))

    print('\nFinished processing anisotropy bij')
    return vals3D, bij, eigValsGrid, eigvecGrid


# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple processReynoldsStress2D(np.ndarray stress_tensor, int realization_iter=0, bint make_anisotropic=0):
    """
    Calculate anisotropy tensor bij, its eigenvalues and eigenvectors from given 2D Reynold stress.
    
    :param stress_tensor: 
    :type stress_tensor: 
    :param realization_iter: 
    :type realization_iter: 
    :param make_anisotropic: 
    :type make_anisotropic: 
    :return: 
    :rtype: 
    """

    # If ndim is not provided but np.float_t is provided, 1D is assumed
    cdef np.ndarray[np.float_t] k, eigval_0, eigval_i
    cdef np.ndarray[np.float_t, ndim = 2] eigvec_0, eigvec_i
    cdef np.ndarray bij, eigval, eigvec
    cdef tuple shape_old
    cdef int i, milestone
    cdef double progress

    shape_old = stress_tensor.shape
    # If Reynolds stress is 4D, then assume first 2D are mesh grid and last 2D are 3 x 3 and reshape to n_cells x 9
    if len(shape_old) == 4:
        stress_tensor = stress_tensor.reshape((shape_old[0]*shape_old[1], 9))
    # Else if Reynolds stress is 3D
    elif len(shape_old) == 3:
        # If 3rd D has 3, then assume nPoint x 3 x 3 and reshape to nPoint x 9
        if stress_tensor.shape[2] == 3:
            stress_tensor = stress_tensor.reshape((stress_tensor.shape[0], 9))
        # Else if 3rd D has 6, then assume nX x nY x 6 and reshape to nPoint x 9
        elif stress_tensor.shape[2] == 6:
            stress_tensor = stress_tensor.reshape((stress_tensor.shape[0]*stress_tensor.shape[1], 6))
        # Else if 3rd D has 9, then assume nX x nY x 9 and rehsape to nPoint x 9
        elif stress_tensor.shape[2] == 9:
            stress_tensor = stress_tensor.reshape((stress_tensor.shape[0]*stress_tensor.shape[1], 9))

    # If stress_tensor was not anisotropic already
    if make_anisotropic:
        # TKE
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        k = 0.5*(stress_tensor[:, 0] + stress_tensor[:, 3] + stress_tensor[:, 5])
        # Avoid FPE
        k[k < 1e-8] = 1e-8
        # Convert Rij to bij
        for i in range(6):
            stress_tensor[:, i] = stress_tensor[:, i]/(2.*k) - 1/3. if i in (0, 3, 5) else stress_tensor[:, i]/(2.*k)

        # Add each anisotropy tensor to each mesh grid location, in depth
        # bij is 3D with z being b11, b12, b13, b21, b22, b23...
        bij = np.dstack((stress_tensor[:, 0], stress_tensor[:, 1], stress_tensor[:, 2],
                             stress_tensor[:, 1], stress_tensor[:, 3], stress_tensor[:, 4],
                             stress_tensor[:, 2], stress_tensor[:, 4], stress_tensor[:, 5]))
        # Use tensor.shape[1] because Numpy dstack 1D array as 1 x N x 1
        bij = bij.reshape((bij.shape[1], 9))
    else:
        bij = stress_tensor

    for i in range(realization_iter):
        print('\nApplying realizability filter ' + str(i + 1))
        bij = makeRealizable(bij)

    # Reshape the z dir to 3x3 instead of 9x1
    # Now bij is 4D, with x, y being nRow, 1, z1, z2 being the 3x3 tensor at (x, y)
    bij = bij.reshape((bij.shape[0], 1, 3, 3))
    # Evaluate eigenvalues and eigenvectors of the symmetric tensor
    # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
    # Each col of eigvec is a vector, thus 3 x 3
    eigval_0, eigvec_0 = np.linalg.eigh(bij[0, 0, :, :])
    eigval_0, eigvec_0 = np.flipud(eigval_0), np.fliplr(eigvec_0)
    # eigvec is nX x nY x 9, where 9 is the flattened eigenvector matrix from np.linalg.eigh()
    # eigval, eigvec = np.zeros(3), np.zeros((bij.shape[0], 1, 9))
    eigval, eigvec = np.empty((bij.shape[0], 3)), np.empty((bij.shape[0], 1, 9))
    eigval[0, :], eigvec[0, 0, :] = eigval_0, eigvec_0.ravel()
    # For gauging progress
    milestone = 25
    # Go through each grid point
    # prange requires nogil that doesn't support python array slicing, and tuple
    for i in range(1, bij.shape[0]):
        # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
        # Each col of eigvec is a vector, thus 3 x 3
        eigval_i, eigvec_i = np.linalg.eigh(bij[i, 0, :, :])
        eigval_i, eigvec_i = np.flipud(eigval_i), np.fliplr(eigvec_i)
        # Each eigval_i is a row, stack them vertically
        # Each eigvec_i is a 3 x 3 matrix, stack them in z dir to each of their i, j = 0 location
        eigval[i, :] = eigval_i
        eigvec[i, 0, :] = eigvec_i.ravel()

        # Gauge progress
        progress = float(i)/(bij.shape[0] + 1)*100.
        if progress >= milestone:
            printf(' %d %%... ', milestone)
            milestone += 25

    # # For gauging progress
    # milestone = 10
    # for i in range(bij.shape[0]):
    #     # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
    #     # Each col of eigvec is a vector, thus 3 x 3
    #     eigVals, eigvec = np.linalg.eigh(bij[i, 0, :, :])
    #     eigVals, eigvec = np.flipud(eigVals), np.fliplr(eigvec)
    #     # Each eigVals is a row, stack them vertically
    #     # Each eigvec is a 3 x 3 matrix, stack them in z dir to each of their i, j = 0 location
    #     eigval = np.vstack((eigval, eigVals))
    #     eigvec[i, 0, :] = eigvec.ravel()
    #     # Gauge progress
    #     progress = float(i)/(bij.shape[0] + 1)*100.
    #     if progress >= milestone:
    #         print(' ' + str(milestone) + '%...')
    #         milestone += 10

    # Reshape eigval to nRow x 1 x 3
    # so that each mesh grid location has 3 eigenvalues
    # Remove the first row since it was dummy
    # Also reshape eigvec from nRow x 1 x 9 to nRow x 1 x 3 x 3
    # so that each col of the 3 x 3 matrix is an eigenvector corresponding to an eigenvalue
    eigval = np.reshape(eigval, (bij.shape[0], 1, 3))
    # eigval = np.reshape(eigval[1:], (bij.shape[0], 1, 3))
    eigvec = np.reshape(eigvec, (bij.shape[0], 1, 3, 3))

    print('\nFinished processing anisotropy bij')
    return bij, eigval, eigvec


# Don't check for array bounds
@cython.boundscheck(False)
# Deactivate negative indexing
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float_t, ndim=2] makeRealizable(np.ndarray[np.float_t, ndim=2] labels):
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
    cdef np.ndarray[np.float_t, ndim=2] A, evectors
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple evaluateInvariantBasisCoefficients(np.ndarray tb, np.ndarray bij, double cap = 100.):
    cdef np.ndarray[np.float_t, ndim = 2] tb_p, g
    cdef np.ndarray[np.float_t] rmse
    cdef int p, verboseInterval, milestone
    print('\nEvaluating invariant basis coefficients g by least squares fitting at each point...')
    # Initialize tensor basis coefficients, nPoint x nBasis
    g, rmse = np.empty((tb.shape[0], tb.shape[1])), np.empty(tb.shape[0])
    # Gauge progress
    verboseInterval, milestone = int(tb.shape[0]/10.), 0
    # Go through each point
    for p in range(tb.shape[0]):
        # Do the same as above, just for each point
        # Row being 9 components and column being nBasis
        # For each point p, tb[p].T is 9 component x nBasis, bij[p] shape is (9,)
        tb_p = tb[p].T
        g[p] = np.linalg.lstsq(tb_p, bij[p], None)[0]
        # TODO: couldn't get RMSE driectly from linalg.lstsq cuz my rank of tb[p] is 5 < 9?
        rmse[p] = sqrt(np.mean(np.square(bij[p] - np.dot(tb_p, g[p]))))
        # Gauge progress
        if p%verboseInterval == 0:
            print('{}%...'.format(milestone))
            milestone += 10

    # Advanced slicing is not support by nijt
    # Cap extreme values
    g[g > cap] = cap
    g[g < -cap] = cap

    return g, rmse
