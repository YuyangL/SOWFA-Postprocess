import numpy as np
from Utilities import timer
from numba import jit, njit, prange
from scipy.interpolate import griddata
import os
import PostProcess_AnisotropyTensor

class SliceProperties:
    def __init__(self, time = 'latest', caseDir = '/media/yluan/Toshiba External Drive', caseName = 'ALM_N_H_ParTurb', caseSubfolder = 'Slices', resultFolder = 'Result', xOrientate = 0):
        self.caseFullPath = caseDir + '/' + caseName + '/' + caseSubfolder + '/'
        self.resultPath = self.caseFullPath + resultFolder + '/'
        # If time is 'latest', find it automatically, excluding the result folder
        self.time = os.listdir(self.caseFullPath)[-2] if time is 'latest' else str(time)
        # Update case full path to include time folder as well
        self.caseFullPath += self.time + '/'
        # Add the selected time folder in result path if not existent already
        self.resultPath += self.time + '/'
        try:
            os.makedirs(self.resultPath)
        except OSError:
            pass

        # Orientate x in the x-y plane in case of angled flow direction
        # Angle in rad and counter-clockwise
        self.xOrientate = xOrientate


    @timer
    # Numba prange doesn't support dict
    # @jit(parallel = True, fastmath = True)
    def readSlices(self, propertyNames = ('U',), sliceNames = ('alongWind',), sliceNamesSub = 'Slice', skipCol = 3, skipRow = 0, fileExt = 'raw'):
        # First 3 columns are x, y, z, thus skipCol = 3
        # skipRow unnecessary since np.genfromtxt trim any header with # at front
        # self.sliceNames need to mutable
        # self.sliceNames = list((sliceNames,)) if isinstance(sliceNames, str) else list(sliceNames)
        sliceNames = (sliceNames,) if isinstance(sliceNames, str) else sliceNames
        self.propertyNames = (propertyNames,) if isinstance(propertyNames, str) else propertyNames
        # Combine propertyName with sliceNames and Subscript to form the full file names
        # Don't know why I had to copy it...
        # self.sliceNames = ['placeholder']*len(sliceNames)*len(propertyNames)
        self.sliceNames = []
        for propertyName in self.propertyNames:
            for sliceName in sliceNames:
                self.sliceNames.append(propertyName + '_' + sliceName)

        self.slicesVal, self.slicesOrientate, self.slicesCoor = {}, {}, {}
        # Go through all specified slices
        # and append coordinates,, slice type (vertical or horizontal), and slice values each to dictionaries
        # Keys are slice names
        for sliceName in self.sliceNames:
            vals = np.genfromtxt(self.caseFullPath + sliceName + '_' + sliceNamesSub + '.' + fileExt)
            # If max(z) - min(z) < 1 then it's assumed horizontal
            # partition('.') removes anything after '.'
            # fileName.partition('.')[0]
            self.slicesOrientate[sliceName] = 'vertical' if (vals[skipRow:, 2]).max() - (
                vals[skipRow:, 2]).min() > 1. else 'horizontal'
            # # If interpolation enabled
            # if interpMethod not in ('none', 'None'):
            #     print('\nInterpolation enabled')
            #     # X2D, Y2D, Z2D, vals2D dictionary after interpolation
            #     slicesX[fileName.partition('.')[0]], slicesY[fileName.partition('.')[0]], slicesZ[fileName.partition('.')[0]], slicesVal[fileName.partition('.')[0]] = \
            #         self.interpolateSliceData(vals[skipRow:, 0], vals[skipRow:, 1], vals[skipRow:, 2], vals[skipRow:, skipCol:], sliceOrientate = slicesOrientate[fileName.partition('.')[0]], precisionX = precision[0], precisionY = precision[1], precisionZ = precision[2], interpMethod = interpMethod)
            #
            # else:
            # X, Y, Z coordinate dictionary without interpolation
            self.slicesCoor[sliceName] = vals[skipRow:, :skipCol]
            # Vals dictionary without interpolation
            self.slicesVal[sliceName] = vals[skipRow:, skipCol:]

        print('\n' + str(self.sliceNames) + ' read')
        # return slicesCoor, slicesOrientate, slicesVal


    @staticmethod
    @timer
    @jit(parallel = True, fastmath = True)
    def interpolateDecomposedSliceData_Fast(x, y, z, vals, sliceOrientate = 'vertical', targetCells = 1e4,
                                            xOrientate = \
        0, interpMethod = 'nearest', confineBox = (None,)):
        # confineBox[4:6] will be ignored if slice is horizontal
        # Automatically determine best x, y, z precision
        # If no confine region specified
        if confineBox[0] is None:
            lx = np.max(x) - np.min(x)
            # If vertical slice, ly doesn't contribute to slice interpolation, thus 0
            ly = np.max(y) - np.min(y) if sliceOrientate == 'horizontal' else 0
            # If horizontal slice, z length should be about 0
            lz = np.max(z) - np.min(z) if sliceOrientate == 'vertical' else 0
        else:
            lx = confineBox[1] - confineBox[0]
            ly = confineBox[3] - confineBox[2] if sliceOrientate == 'horizontal' else 0
            lz = confineBox[5] - confineBox[4] if sliceOrientate == 'vertical' else \
                0

        # Sum of all "contributing" lengths, as scaler
        lxyz = lx + ly + lz
        # Weight of each precision, minimum 0.001
        xratio = np.max((lx/lxyz, 0.001))
        # For vertical slices, the resolution for x is the same as y, i.e. targetCells is shared between x and z only
        yratio = np.max((ly/lxyz, 0.001)) if sliceOrientate == 'horizontal' else 1.
        # If horizontal slice, z cell is aimed at 1
        # zratio = 1 effectively removes the impact of z on horizontal slice resolutions
        zratio = np.max((lz/lxyz, 0.001)) if sliceOrientate == 'vertical' else 1.
        # Base precision
        # Using xratio*base*(yratio*base or zratio*base) = targetCells
        precisionBase = targetCells/(xratio*yratio*zratio)
        # If horizontal slice, targetCells is shared between x and y only
        precisionBase = precisionBase**(1/2.)
        # Derived precision, take ceiling
        precisionX = int(np.ceil(xratio*precisionBase))
        # If vertical slice, precisionY is the same as precisionX
        precisionY = int(np.ceil(yratio*precisionBase)) if sliceOrientate == 'horizontal' else precisionX
        precisionZ = int(np.ceil(zratio*precisionBase)) if sliceOrientate == 'vertical' else 1
        print('\nInterpolating slice with precision ({0}, {1}, {2})...'.format(precisionX, precisionY, precisionZ))
        precisionX *= 1j
        precisionY *= 1j
        precisionZ *= 1j
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1, 1)
        bnd_xTarget = (x.min()*bnd[0], x.max()*bnd[1]) if confineBox[0] is None else confineBox[0], confineBox[1]
        bnd_yTarget = (y.min()*bnd[0], y.max()*bnd[1]) if confineBox[0] is None else confineBox[2], confineBox[3]
        bnd_zTarget = (z.min()*bnd[0], z.max()*bnd[1]) if confineBox[0] is None else confineBox[4], confineBox[5]
        # If vertical slice
        if sliceOrientate == 'vertical':
            # Known x and z coordinates, to be interpolated later
            knownPoints = np.vstack((x, z)).T
            # Interpolate x and z according to precisions
            x2D, z2D = np.mgrid[bnd_xTarget[0]:bnd_xTarget[1]:precisionX, bnd_zTarget[0]:bnd_zTarget[1]:precisionZ]
            # Then interpolate y in the same fashion of x
            # Thus the same precision as x
            y2D, _ = np.mgrid[bnd_yTarget[0]:bnd_yTarget[1]:precisionX, bnd_zTarget[0]:bnd_zTarget[1]:precisionZ]
            # In case the vertical slice is at a negative angle,
            # i.e. when x goes from low to high, y goes from high to low,
            # flip y2D from low to high to high to low
            y2D = np.flipud(y2D) if x[0] > x[1] else y2D
        # Else if slice is horizontal
        else:
            knownPoints = np.vstack((x, y)).T
            x2D, y2D = np.mgrid[bnd_xTarget[0]:bnd_xTarget[1]:precisionX, bnd_yTarget[0]:bnd_yTarget[1]:precisionY]
            # For horizontal slice, z is constant, thus not affected by confineBox
            # Also z2D is like y2D, thus same precision as y
            _, z2D = np.mgrid[bnd_xTarget[0]:bnd_xTarget[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionY]

        # Decompose the vector/tensor of slice values
        # If vector, order is x, y, z
        # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
        # Second axis to interpolate is z if vertical slice otherwise y fo horizontal slices
        gridSecondCoor = z2D if sliceOrientate == 'vertical' else y2D
        # For gauging progress
        milestone = 33
        # If vals is 2D
        if len(vals.shape) == 2:
            # Initialize nRow x nCol x nComponent array vals3D by interpolating first component of the values, x, or xx
            vals3D = np.empty((x2D.shape[0], x2D.shape[1], vals.shape[1]))
            # Then go through the rest components and stack them in 3D
            for i in prange(vals.shape[1]):
                # Each component is interpolated from the known locations pointsXZ to refined fields (x2D, z2D)
                vals3D_i = griddata(knownPoints, vals[:, i].ravel(), (x2D, gridSecondCoor), method = interpMethod)
                # vals3D = np.dstack((vals3D, vals3D_i))
                vals3D[:, :, i] = vals3D_i
                # Gauge progress
                progress = (i + 1)/vals.shape[1]*100.
                if progress >= milestone:
                    print(' {0}%... '.format(milestone))
                    milestone += 33

        # Else if vals is 3D
        else:
            vals3D = np.empty((x2D.shape[0], x2D.shape[1], vals.shape[2]))
            for i in prange(vals.shape[2]):
                vals3D_i = griddata(knownPoints, vals[:, :, i].ravel(), (x2D, gridSecondCoor), method = interpMethod)
                vals3D[:, :, i] = vals3D_i
                progress = (i + 1)/vals.shape[2]*100.
                if progress >= milestone:
                    print(' {0}%... '.format(milestone))
                    milestone += 33

        # if xOrientate != 0:
        #     # If vector, x, y, z
        #     if vals.shape[1] == 3:
        #         vals3D['0'] = vals3D['0']*np.cos(xOrientate) + vals3D['1']*np.sin(xOrientate)
        #         vals3D['1'] = -vals3D['0']*np.sin(xOrientate) + vals3D['1']*np.cos(xOrientate)
        #     else:
        #         vals3D['0'] =

        x2D, y2D, z2D = np.nan_to_num(x2D), np.nan_to_num(y2D), np.nan_to_num(z2D)
        vals3D = np.nan_to_num(vals3D)

        return x2D, y2D, z2D, vals3D


    # [DEPRECATED] Refer to interpolateDecomposedSliceData_Fast()
    @staticmethod
    @timer
    @jit(parallel = True, fastmath = True)
    def interpolateDecomposedSliceData(x, y, z, vals, sliceOrientate = 'vertical', xOrientate = 0, precisionX = 1500j, precisionY = 1500j,
                          precisionZ = 500j, interpMethod = 'cubic'):
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1.00001, 0.99999)
        if sliceOrientate is 'vertical':
            # Known x and z coordinates, to be interpolated later
            knownPoints = np.vstack((x, z)).T
            # Interpolate x and z according to precisions
            x2D, z2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
            # Then interpolate y in the same fashion of x
            y2D, _ = np.mgrid[y.min()*bnd[0]:y.max()*bnd[1]:precisionY, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
            # In case the vertical slice is at a negative angle,
            # i.e. when x goes from low to high, y goes from high to low,
            # flip y2D from low to high to high to low
            y2D = np.flipud(y2D) if x[0] > x[1] else y2D
        else:
            knownPoints = np.vstack((x, y)).T
            x2D, y2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, y.min()*bnd[0]:y.max()*bnd[1]:precisionY]
            _, z2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]

        # Decompose the vector/tensor of slice values
        # If vector, order is x, y, z
        # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
        valsDecomp = {}
        for i in range(vals.shape[1]):
            if sliceOrientate is 'vertical':
                # Each component is interpolated from the known locations pointsXZ to refined fields (x2D, z2D)
                valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2D, z2D), method = interpMethod)
            else:
                valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2D, y2D), method = interpMethod)

        # if xOrientate != 0:
        #     # If vector, x, y, z
        #     if vals.shape[1] == 3:
        #         valsDecomp['0'] = valsDecomp['0']*np.cos(xOrientate) + valsDecomp['1']*np.sin(xOrientate)
        #         valsDecomp['1'] = -valsDecomp['0']*np.sin(xOrientate) + valsDecomp['1']*np.cos(xOrientate)
        #     else:
        #         valsDecomp['0'] =

        return x2D, y2D, z2D, valsDecomp


    @staticmethod
    @timer
    @njit(parallel = True, fastmath = True)
    def processAnisotropyTensor_Fast(vals3D):
        # TKE in the interpolated mesh
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        k = 0.5*(vals3D[:, :, 0] + vals3D[:, :, 3] + vals3D[:, :, 5])
        # Convert Rij to bij
        for i in prange(6):
            vals3D[:, :, i] = vals3D[:, :, i]/(2.*k) - 1/3. if i in (0, 3, 5) else vals3D[:, :, i]/(2.*k)

        # Add each anisotropy tensor to each mesh grid location, in depth
        # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
        tensors = np.dstack((vals3D[:, :, 0], vals3D[:, :, 1], vals3D[:, :, 2],
                             vals3D[:, :, 1], vals3D[:, :, 3], vals3D[:, :, 4],
                             vals3D[:, :, 2], vals3D[:, :, 4], vals3D[:, :, 5]))
        # Reshape the z dir to 3x3 instead of 9x1
        # Now tensors is 4D, with x, y being mesh grid, z1, z2 being the 3x3 tensor at (x, y)
        tensors = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))

        # Evaluate eigenvalues of symmetric tensor
        eigValsGrid = np.zeros((1,3))
        for i in range(tensors.shape[0]):
            for j in range(tensors.shape[1]):
                # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
                eigVals = np.linalg.eigvalsh(tensors[i, j, :, :])
                # Numba njit doesn't support np.flipud, thus manually reverse eigVals to high to low
                tmpVal = eigVals[0]
                eigVals[0] = eigVals[2]
                eigVals[2] = tmpVal
                eigVals2D = eigVals.reshape((1, 3))
                # if i == 0 and j == 0:
                #     eigValsGrid0 = np.zeros_like(eigVals2D)
                #     eigValsGrid = np.vstack((eigValsGrid0, eigVals2D))
                #     print(eigValsGrid.shape)
                # else:
                # Each eigVals is a row, stack them vertically
                eigValsGrid = np.vstack((eigValsGrid, eigVals2D))

        # Reshape eigVals to nRow x nCol x 3
        # so that each mesh grid location has 3 eigenvalues
        # Remove the first row since it was dummy
        eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))

        return vals3D, tensors, eigValsGrid


    # [DEPRECATED] Refer to PostProcess_AnisotropyTensor
    @staticmethod
    @timer
    @njit(parallel = True, fastmath = True)
    def processAnisotropyTensor_Uninterpolated(vals2D):
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
        # Reshape the z dir to 3x3 instead of 9x1
        # Now tensors is 4D, with x, y being nRow, 1, z1, z2 being the 3x3 tensor at (x, y)
        # Don't know why x, y shape has to be flipped?
        # tensors1 = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))
        tensors = tensors.reshape((tensors.shape[1], tensors.shape[0], 3, 3))
        # print(tensors1[0, 100, :])
        # print(tensors[100, 0, :])

        # Evaluate eigenvalues of symmetric tensor
        eigValsGrid = np.zeros((1, 3))
        for i in range(tensors.shape[0]):
            for j in range(tensors.shape[1]):
                # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
                eigVals = np.linalg.eigvalsh(tensors[i, j, :, :])
                # Numba njit doesn't support np.flipud, thus manually reverse eigVals to high to low
                tmpVal = eigVals[0]
                eigVals[0] = eigVals[2]
                eigVals[2] = tmpVal
                eigVals2D = eigVals.reshape((1, 3))
                # if i == 0 and j == 0:
                #     eigValsGrid0 = np.zeros_like(eigVals2D)
                #     eigValsGrid = np.vstack((eigValsGrid0, eigVals2D))
                #     print(eigValsGrid.shape)
                # else:
                # Each eigVals is a row, stack them vertically
                eigValsGrid = np.vstack((eigValsGrid, eigVals2D))

        # Reshape eigVals to nRow x nCol x 3
        # so that each mesh grid location has 3 eigenvalues
        # Remove the first row since it was dummy
        eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))

        return vals2D, tensors, eigValsGrid


    @staticmethod
    @timer
    @jit(parallel = True, fastmath = True)
    def processAnisotropyTensor(valsDecomp):
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
        eigValsGrid = [0, 0, 0]
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


    @staticmethod
    @timer
    # @njit(parallel = True, fastmath = True)
    def getBarycentricMapCoordinates(eigValsGrid, c_offset = 0.65, c_exp = 5.):
        # Coordinates of the anisotropy tensor in the tensor basis {a1c, a2c, a3c}. From Banerjee (2007),
        # C1c = lambda1 - lambda2,
        # C2c = 2(lambda2 - lambda3),
        # C3c = 3lambda3 + 1
        c1 = eigValsGrid[:, :, 0] - eigValsGrid[:, :, 1]
        # Not used for coordinates, only for color maps
        c2 = 2*(eigValsGrid[:, :, 1] - eigValsGrid[:, :, 2])
        c3 = 3*eigValsGrid[:, :, 2] + 1
        # Corners of the barycentric triangle
        # Can be random coordinates?
        x1c, x2c, x3c = 1., 0., 1/2.
        y1c, y2c, y3c = 0, 0, np.sqrt(3)/2
        # xBary, yBary = c1*x1c + c2*x2c + c3*x3c, c1*y1c + c2*y2c + c3*y3c
        xBary, yBary = c1 + 0.5*c3, y3c*c3
        # Origin RGB values
        rgbVals = np.dstack((c1, c2, c3))
        # For better barycentric map, use transformation on c1, c2, c3, as in Emory et al. (2014)
        # ci_star = (ci + c_offset)^c_exp,
        # Improved RGB = [c1_star, c2_star, c3_star]
        rgbValsNew = np.empty((c1.shape[0], c1.shape[1], 3))
        # Each 3rd dim is an RGB array of the 2D grid
        for i in prange(3):
            rgbValsNew[:, :, i] = (rgbVals[:, :, i] + c_offset)**c_exp

        return xBary, yBary, rgbValsNew


    @staticmethod
    @timer
    @jit
    def mergeHorizontalComponents(valsDecomp):
        valsDecomp['hor'] = np.sqrt(valsDecomp['0']**2 + valsDecomp['1']**2)
        return valsDecomp


    @staticmethod
    @timer
    @jit(parallel = True, fastmath = True)
    def calcSliceMeanDissipationRate(epsilonSGSmean, nuSGSmean, nu = 1e-5, save = False, resultPath = '.'):
        # According to Eq 5.64 - Eq 5.68 of Sagaut (2006), for isotropic homogeneous turbulence,
        # <epsilon> = <epsilon_resolved> + <epsilon_SGS>,
        # <epsilon_resolved>/<epsilon_SGS> = 1/(1 + (<nu_SGS>/nu)),
        # where epsilon is the total turbulence dissipation rate (m^2/s^3); and <> is statistical averaging
        epsilonMean = epsilonSGSmean/(1 - (1/(1 + nuSGSmean/nu)))

        if save:
            pickle.dump(epsilonMean, open(resultPath + '/epsilonMean.p', 'wb'))
            print('\nepsilonMean saved at {0}'.format(resultPath))

        return epsilonMean


    @staticmethod
    @timer
    # Advanced slicing not supported by njit, e.g. Sij[Sij > cap] - ...
    @jit(parallel = True, fastmath = True)
    def calcSliceSijAndRij(grad_u, tke, eps, cap = 7.):
        """
        Calculates the non-dimonsionalized strain rate and rotation rate tensors.  Normalizes by k and eps:
        Sij = k/eps * 0.5* (grad_u  + grad_u^T)
        Rij = k/eps * 0.5* (grad_u  - grad_u^T)
        :param grad_u: num_points X 3 X 3
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate epsilon
        :param cap: This is the max magnitude that Sij or Rij components are allowed.  Greater values
                    are capped at this level
        :return: Sij, Rij: num_points X 3 X 3 tensors
        """

        if len(grad_u.shape) == 2 and grad_u.shape[1] == 9:
            grad_u = grad_u.reshape((grad_u.shape[0], 3, 3))

        if len(tke.shape) == 2:
            tke = tke.ravel()

        if len(eps.shape) == 2:
            eps = eps.ravel()

        num_points = grad_u.shape[0]
        eps = np.maximum(eps, 1e-8)
        tke_eps = tke/eps
        Sij = np.zeros((num_points, 3, 3))
        Rij = np.zeros((num_points, 3, 3))
        for i in prange(num_points):
            Sij[i, :, :] = tke_eps[i]*0.5*(grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            Rij[i, :, :] = tke_eps[i]*0.5*(grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        maxSij, maxRij = np.amax(Sij), np.amax(Rij)
        minSij, minRij = np.amin(Sij), np.amin(Rij)
        print(' Max of Sij is ' + str(maxSij) + ', and of Rij is ' + str(maxRij))
        print(' Min of Sij is ' + str(minSij) + ', and of Rij is ' + str(minRij))
        # Why caps?????????????
        Sij[Sij > cap] = cap
        Sij[Sij < -cap] = -cap
        Rij[Rij > cap] = cap
        Rij[Rij < -cap] = -cap

        # Because we enforced limits on maximum Sij values, we need to re-enforce trace of 0
        for i in prange(num_points):
            Sij[i, :, :] = Sij[i, :, :] - 1/3.*np.eye(3)*np.trace(Sij[i, :, :])

        return Sij, Rij
    
    
    @staticmethod
    @timer
    # Numba is unable to determine "self" type
    @jit(parallel = True, fastmath = True)
    def calcSliceScalarBasis(Sij, Rij, is_train = False, cap = 2.0, is_scale = True, mean = None, std = None):
        """
        Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
        this returns a set of normalized scalar invariants
        :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
        :param Rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        :return: mean: Mean of the scalar basis, can be used when predicting
        :return: std: Standard deviation of the scalar basis, can be used when predicting
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3) * 2.0
        >>> B[0, 1, 0] = 1.0
        >>> B[0, 0, 1] = -1.0
        >>> tdp = PostProcess_SliceData()
        >>> scalar_basis, mean, std = tdp.calcSliceScalarBasis(A, B, is_scale=False)
        >>> print scalar_basis
        [[ 12.  -2.  24.  -4.  -8.]]
        """
        if is_train or mean is None or std is None:
            print("Re-setting normalization constants")
            
        num_points = Sij.shape[0]
        num_invariants = 5
        invariants = np.zeros((num_points, num_invariants))
        for i in prange(num_points):
            invariants[i, 0] = np.trace(np.dot(Sij[i, :, :], Sij[i, :, :]))
            invariants[i, 1] = np.trace(np.dot(Rij[i, :, :], Rij[i, :, :]))
            invariants[i, 2] = np.trace(np.dot(Sij[i, :, :], np.dot(Sij[i, :, :], Sij[i, :, :])))
            invariants[i, 3] = np.trace(np.dot(Rij[i, :, :], np.dot(Rij[i, :, :], Sij[i, :, :])))
            invariants[i, 4] = np.trace(
                np.dot(np.dot(Rij[i, :, :], Rij[i, :, :]), np.dot(Sij[i, :, :], Sij[i, :, :])))

        # Renormalize invariants using mean and standard deviation:
        if is_scale:
            if mean is None or std is None:
                is_train = True

            if is_train:
                mean = np.zeros((num_invariants, 2))
                std = np.zeros((num_invariants, 2))
                mean[:, 0] = np.mean(invariants, axis = 0)
                std[:, 0] = np.std(invariants, axis = 0)

            invariants = (invariants - mean[:, 0])/std[:, 0]
            maxInvariants, minInvariants = np.amax(invariants), np.amin(invariants)
            print(' Max of scaled scalar basis is {}'.format(maxInvariants))
            print(' Min of scaled scalar basis is {}'.format(minInvariants))
            # Why cap?????
            invariants[invariants > cap] = cap  # Cap max magnitude
            invariants[invariants < -cap] = -cap
            invariants = invariants*std[:, 0] + mean[:, 0]
            if is_train:
                mean[:, 1] = np.mean(invariants, axis = 0)
                std[:, 1] = np.std(invariants, axis = 0)

            invariants = (invariants - mean[:, 1])/std[:, 1]  # Renormalize a second time after capping
        return invariants, mean, std


    @staticmethod
    @timer
    @njit(parallel = True, fastmath = True)
    def calcSliceTensorBasis(Sij, Rij, quadratic_only = False, is_scale = True):
        """
        Given non-dimsionalized Sij and Rij, it calculates the tensor basis
        :param Sij: normalized strain rate tensor
        :param Rij: normalized rotation rate tensor
        :param quadratic_only: True if only linear and quadratic terms are desired.  False if full basis is desired.
        :return: T_flat: num_points X num_tensor_basis X 9 numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3)
        >>> B[0, 1, 0] = 3.0
        >>> B[0, 0, 1] = -3.0
        >>> tdp = PostProcess_SliceData()
        >>> tb = tdp.calcSliceTensorBasis(A, B, is_scale=False)
        >>> print tb[0, :, :]
        [[  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -3.   0.   0.   0.  -3.   0.   0.   0.   6.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -6.   0.   0.   0.  -6.   0.   0.   0.  12.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -6.   0.   0.   0.  -6.   0.   0.   0.  12.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]]
        """
        num_points = Sij.shape[0]
        if not quadratic_only:
            num_tensor_basis = 10
        else:
            num_tensor_basis = 4

        T = np.zeros((num_points, num_tensor_basis, 3, 3))
        for i in prange(num_points):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            T[i, 0, :, :] = sij
            T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
            T[i, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
            T[i, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
            if not quadratic_only:
                T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
                T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) \
                                + np.dot(sij, np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
                T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
                T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
                T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) \
                                + np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
                T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                                - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
            # Enforce zero trace for anisotropy
            for j in range(num_tensor_basis):
                T[i, j, :, :] = T[i, j, :, :] - 1./3.*np.eye(3)*np.trace(T[i, j, :, :])

        # Scale down to promote convergence
        if is_scale:
            scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
            for i in prange(num_tensor_basis):
                T[:, i, :, :] /= scale_factor[i]

        # Flatten:
        # T_flat = np.zeros((num_points, num_tensor_basis, 9))
        # for i in prange(3):
        #     for j in range(3):
        #         T_flat[:, :, 3*i+j] = T[:, :, i, j]
        T_flat = T.reshape((T.shape[0], T.shape[1], 9))

        return T_flat


    @staticmethod
    @timer
    def rotateSpatialCorrelationTensors(listData, rotateXY = 0., rotateUnit = 'rad', dependencies = ('xx',)):
        """
        Rotate one or more single/double spatial correlation tensor field/slice data in the x-y plane, doesn't work on rate of strain/rotation tensors
        :param listData: Any (nPt x nComponent) or (nX x nY x nComponent) or (nX x nY x nZ x nComponent) data of interest, appended to a tuple/list.
        If nComponent is 6, data is symmetric 3 x 3 double spatial correlation tensor field.
        If nComponent is 9, data is single/double spatial correlation tensor field depending on dependencies keyword
        :type listData: tuple/list([:, :]/[:, :, :]/[:, :, :, :])
        :param rotateXY:
        :type rotateXY:
        :param rotateUnit:
        :type rotateUnit:
        :param dependencies: 'x' or 'xx'. Default is ('xx',)
        Whether the component of data is dependent on single spatial correlation 'x' e.g. gradient, vector, or double spatial correlation 'xx' e.g. double correlation.
        Only used if nComponent is 9
        :type dependencies: str or tuple/list(str)
        :return: listData_rot
        :rtype:
        """

        # Ensure list input since each listData[i] is modified to nPt x nComponent later
        listData = list((listData,)) if isinstance(listData, np.ndarray) else list(listData)
        # Ensure tuple input
        dependencies = (dependencies,) if isinstance(dependencies, str) else dependencies
        # Ensure dependencies has the same entries as the number of data provided
        dependencies *= len(listData) if len(dependencies) < len(listData) else 1
        # Ensure radian unit
        rotateXY *= np.pi/180 if rotateUnit != 'rad' else 1.

        # Reshape here screwed up njit :/
        @jit(parallel = True, fastmath = True)
        def __transform(listData, rotateXY, rotateUnit, dependencies):
            # Copy listData (a list) that has original shapes as listData will be flattened to nPt x nComponent
            listDataRot_oldShapes = listData.copy()
            sinVal, cosVal = np.sin(rotateXY), np.cos(rotateXY)
            # Go through every data in listData and flatten to nPt x nComponent if necessary
            for i in prange(len(listData)):
                # # Ensure Numpy array
                # listData[i] = np.array(listData[i])
                # Flatten data from 3D to 2D
                if len(listData[i].shape) >= 3:
                    # Go through 2nd D to (last - 1) D
                    nRow = 1
                    for j in range(len(listData[i].shape) - 1):
                        nRow *= listData[i].shape[j]

                    # Flatten data to nPt x nComponent
                    nComponent = listData[i].shape[len(listData[i].shape) - 1]
                    listData[i] = listData[i].reshape((nRow, nComponent))

            # Create a copy so listData values remain unchanged during the transformation, listData[i] is nPt x nComponent now
            listData_rot = listData.copy()
            # Go through all provided data and perform transformation
            for i in prange(len(listData)):
                # Number of component is the last D
                nComponent = listData[i].shape[len(listData[i].shape) - 1]
                # If nComponent is 3, i.e. x, y, z or data is single spatial correlation with 9 components
                if nComponent == 3 or (nComponent == 9 and dependencies[i] == 'x'):
                    # x_rot becomes x*cos + y*sin
                    x_rot = listData[i][:, 0]*cosVal + listData[i][:,
                                                       1]*sinVal
                    # y_rot becomes -x*sin + y*cos
                    y_rot = -listData[i][:, 0]*sinVal + listData[i][:,
                                                        1]*cosVal
                    # z_rot doesn't change
                    z_rot = listData[i][:, 2]
                    listData_rot[i][:, 0], listData_rot[i][:, 1], listData_rot[i][:, 2] = x_rot, y_rot, z_rot
                    # If 9 components with single spatial correlation, e.g.gradient tensor, do it for 2nd row and 3rd row
                    if nComponent == 9:
                        x_rot2 = listData[i][:, 3]*cosVal + listData[i][:,
                                                            4]*sinVal
                        y_rot2 = -listData[i][:, 3]*sinVal + listData[i][:,
                                                            4]*cosVal
                        z_rot2 = listData[i][:, 5]
                        x_rot3 = listData[i][:, 6]*cosVal + listData[i][:,
                                                            7]*sinVal
                        y_rot3 = -listData[i][:, 6]*sinVal + listData[i][:,
                                                             7]*cosVal
                        z_rot3 = listData[i][:, 8]
                        listData_rot[i][:, 3], listData_rot[i][:, 4], listData_rot[i][:, 5] = x_rot2, y_rot2, z_rot2
                        listData_rot[i][:, 6], listData_rot[i][:, 7], listData_rot[i][:, 8] = x_rot3, y_rot3, z_rot3

                # Else if nComponent is 6 or 9 with double spatial correlation
                elif nComponent == 6 or (nComponent == 9 and dependencies[i] == 'xx'):
                    # If 6 components and double spatial correlation, i.e.xx, xy, xz, yy, yz, zz
                    # or 9 components and double spatial correlation, i.e. xx, xy, xz, yx, yy, yz, zx, zy, zz
                    if dependencies[i] == 'xx':
                        xx, xy, xz = listData[i][:, 0], listData[i][:, 1], listData[i][:, 2]
                        yy, yz = listData[i][:, 3], listData[i][:, 4]
                        zz = listData[i][:, 5]
                        # xx_rot becomes x_rot*x_rot = xx*cos^2 + 2xy*sin*cos + yy*sin^2
                        xx_rot = xx*cosVal**2 + 2*xy*sinVal*cosVal + yy*sinVal**2
                        # xy_rot becomes x_rot*y_rot = xy*cos^2 + yy*sin*cos - xx*sin*cos -xy*sin^2
                        xy_rot = xy*cosVal**2 + yy*sinVal*cosVal - xx*sinVal*cosVal - xy*sinVal**2
                        # xz_rot become x_rot*z_rot = xz*cos + yz*sin
                        xz_rot = xz*cosVal + yz*sinVal
                        # yy_rot becomes y_rot*y_rot = yy*cos^2 - 2xy*sin*cos + xx*sin^2
                        yy_rot = yy*cosVal**2 - 2*xy*sinVal*cosVal + xx*sinVal**2
                        # yz_rot becomes y_rot*z_rot = yz*cos - xz*sin
                        yz_rot = yz*cosVal - xz*sinVal
                        # zz_rot remains the same
                        zz_rot = zz
                        # Apply these changes
                        listData_rot[i][:, 0], listData_rot[i][:, 1], listData_rot[i][:, 2] = xx_rot, xy_rot, xz_rot
                        # For 6 component symmetric data
                        if nComponent == 6:
                            listData_rot[i][:, 3], listData_rot[i][:, 4] = yy_rot, yz_rot
                            listData_rot[i][:, 5] = zz_rot
                        # For 9 component symmetric data
                        elif nComponent == 9:
                            listData_rot[i][:, 3], listData_rot[i][:, 4], listData_rot[i][:, 5] = xy_rot, yy_rot, yz_rot
                            listData_rot[i][:, 6], listData_rot[i][:, 7], listData_rot[i][:, 8] = xz_rot, yz_rot, zz_rot

                # Lastly, reshape transformed data i back to old shape while replacing old values with the transformed one
                listDataRot_oldShapes[i] = listData_rot[i].reshape(np.array(listDataRot_oldShapes[i]).shape)
            return listDataRot_oldShapes

        return __transform(listData, rotateXY, rotateUnit, dependencies)


    def processSliceTBNN_Inputs(self, gradUAvg, kMean, epsilonMean, capSijRij = 1e9, capSB = 1e9, scaleSB = True, scaleTB = True):
        # Calculate the non-dimensionalized Sij and Rij using kMean/epsilonMean
        Sij, Rij = self.calcSliceSijAndRij(gradUAvg, kMean, epsilonMean, capSijRij)
        # Calculate the 5 scalar basis using Sij and Rij
        scalarBasis, mean, std = self.calcSliceScalarBasis(Sij, Rij, is_train = True, is_scale = scaleSB, cap = capSB)
        # Calculate the 10 tensor basis using Sij and Rij
        tensorBasis = self.calcSliceTensorBasis(Sij, Rij, scaleTB)

        return scalarBasis, tensorBasis, mean, std












if __name__ is '__main__':
    import time as t
    from PlottingTool import PlotSurfaceSlices3D
    import matplotlib.pyplot as plt
    from scipy import ndimage
    import pickle

    """
    User Inputs
    """
    time = 'latest'
    caseDir = 'J:'
    caseDir = '/media/yluan/1'
    caseName = 'ALM_N_H_ParTurb'
    propertyName = 'uuPrime2'
    sliceNames = 'alongWindRotorOne'
    # Orientation of x-axis in x-y plane, in case of angled flow direction
    # Only used for values decomposition
    # Angle in rad and counter-clockwise
    xOrientate = 6/np.pi
    precisionX, precisionY, precisionZ = 1000j, 1000j, 333j
    interpMethod = 'nearest'
    plot = 'bary'

    case = SliceProperties(time = time, caseDir = caseDir, caseName = caseName, xOrientate = xOrientate)

    case.readSlices(propertyName = propertyName, sliceNames = sliceNames)

    for sliceName in case.sliceNames:
        """
        Process Uninterpolated Anisotropy Tensor
        """
        vals2D = case.slicesVal[sliceName]
        # x2D, y2D, z2D, vals3D = \
        #     case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], case.slicesVal[sliceName], sliceOrientate = case.slicesOrientate[sliceName], xOrientate = case.xOrientate, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)

        # Another implementation of processAnisotropyTensor() in Cython
        t0 = t.time()
        # vals3D, tensors, eigValsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor(vals3D)
        # In each eigenvector 3 x 3 matrix, 1 col is a vector
        vals2D, tensors, eigValsGrid, eigVecsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor_Uninterpolated(vals2D, realizeIter = 0)
        t1 = t.time()
        ticToc = t1 - t0

        # valsDecomp, tensors, eigValsGrid = case.processAnisotropyTensor_Fast(valsDecomp)
        # vals2D, tensors, eigValsGrid = case.processAnisotropyTensor_Uninterpolated(vals2D)


        """
        Interpolation
        """
        if plot == 'bary':
            xBary, yBary, rgbVals = case.getBarycentricMapCoordinates(eigValsGrid)

            x2D, y2D, z2D, rgbVals = \
                case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], rgbVals, sliceOrientate =
                                                         case.slicesOrientate[sliceName], xOrientate = case.xOrientate,
                                                         precisionX = precisionX, precisionY =
                                                         precisionY, precisionZ = precisionZ, interpMethod = interpMethod)

        elif plot == 'quiver':
            x2D, y2D, z2D, eigVecs3D = \
                case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1],
                                                         case.slicesCoor[sliceName][:, 2], eigVecsGrid[:, :, :, 0], sliceOrientate =
                                                         case.slicesOrientate[sliceName], xOrientate = case.xOrientate,
                                                         precisionX = precisionX, precisionY =
                                                         precisionY, precisionZ = precisionZ, interpMethod = interpMethod)


        """
        Plotting
        """
        if plot == 'bary':
            print('\nDumping values...')
            pickle.dump(tensors, open(case.resultPath + sliceName + '_rgbVals.p', 'wb'))
            pickle.dump(x2D, open(case.resultPath + sliceName + '_x2D.p', 'wb'))
            pickle.dump(y2D, open(case.resultPath + sliceName + '_y2D.p', 'wb'))
            pickle.dump(z2D, open(case.resultPath + sliceName + '_z2D.p', 'wb'))

            # # Custom RGB colormap, with only red, green, and blue
            # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            # nBin = 1000
            # cmapName = 'rgb'
            # from matplotlib.colors import LinearSegmentedColormap
            # rgbCm = LinearSegmentedColormap.from_list(
            #         cmapName, colors, N = nBin)

            # rgbVals[rgbVals < 0] = 0.
            # rgbVals[rgbVals > 1] = 1.

            # Rotate the figure 90 deg clockwise
            rgbValsRot = ndimage.rotate(rgbVals, 90)
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(rgbValsRot, origin = 'upper', aspect = 'equal', extent = (0, 3000, 0, 3000))
            # fig.colorbar(im, ax = ax, extend = 'both')
            plt.savefig('R:/bary.png', dpi = 1000)

            # baryPlot = PlotSurfaceSlices3D(x2D, y2D, z2D, (0,), show = True, name = 'bary', figDir = 'R:', save = True)
            # baryPlot.cmapLim = (0, 1)
            # baryPlot.cmapNorm = rgbVals
            # # baryPlot.cmapVals = plt.cm.ScalarMappable(norm = rgbVals, cmap = None)
            # baryPlot.cmapVals = rgbVals
            # # baryPlot.cmapVals.set_array([])
            # baryPlot.plot = baryPlot.cmapVals
            # baryPlot.initializeFigure()
            # baryPlot.axes[0].plot_surface(x2D, y2D, z2D, cstride = 1, rstride = 1, facecolors = rgbVals, vmin = 0, vmax = 1, shade = False)
            # baryPlot.finalizeFigure()
            #
            #
            #
            #
            #
            # print('\nDumping values...')
            # pickle.dump(tensors, open(case.resultPath + sliceName + '_tensors.p', 'wb'))
            # pickle.dump(case.slicesCoor[sliceName][:, 0], open(case.resultPath + sliceName + '_x.p', 'wb'))
            # pickle.dump(case.slicesCoor[sliceName][:, 1], open(case.resultPath + sliceName + '_y.p', 'wb'))
            # pickle.dump(case.slicesCoor[sliceName][:, 2], open(case.resultPath + sliceName + '_z.p', 'wb'))
            #
            # print('\nExecuting RGB_barycentric_colors_clean...')
            # import RBG_barycentric_colors_clean
            #
            #
            #
            #
            # # valsDecomp = case.mergeHorizontalComponents(valsDecomp)
        elif plot == 'quiver':
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            ax.quiver(x2D, y2D, z2D, eigVecs3D[:, :, 0], eigVecs3D[:, :, 1], eigVecs3D[:, :, 2], length = 0.1, normalize = False)

