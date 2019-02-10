import numpy as np
from Utilities import timer
from numba import jit, njit
from scipy.interpolate import griddata
import os
import PostProcess_AnisotropyTensor

class SliceProperties:
    def __init__(self, time = 'latest', caseDir = '/media/yluan/Toshiba External Drive', caseName = 'ALM_N_H_ParTurb', caseSubfolder = 'Slices', resultFolder = 'Result', xOrientate = 0):
        self.caseFullPath = caseDir + '/' + caseName + '/' + caseSubfolder + '/'
        self.resultPath = self.caseFullPath + resultFolder + '/'
        # Try to make the result folder under caseSubfolder
        try:
            os.mkdir(self.resultPath)
        except OSError:
            pass

        # If time is 'latest', find it automatically, excluding the result folder
        self.time = os.listdir(self.caseFullPath)[-2] if time is 'latest' else str(time)
        # Update case full path to include time folder as well
        self.caseFullPath += self.time + '/'
        # Add the selected time folder in result path if not existent already
        self.resultPath += self.time + '/'
        try:
            os.mkdir(self.resultPath)
        except OSError:
            pass

        # Orientate x in the x-y plane in case of angled flow direction
        # Angle in rad and counter-clockwise
        self.xOrientate = xOrientate


    @timer
    @jit
    def readSlices(self, propertyName = 'U', sliceNames = ('alongWind',), sliceNamesSub = '_Slice', skipCol = 3, skipRow = 0, fileExt = '.raw'):
        # First 3 columns are x, y, z, thus skipCol = 3
        # skipRow unnecessary since np.genfromtxt trim any header with # at front
        self.sliceNames = [sliceNames] if isinstance(sliceNames, str) else sliceNames
        # Combine propertyName with sliceNames and Subscript to form the full file names
        # Don't know why I had to copy it...
        # self.fileNames = list(sliceNames).copy()
        for i, name in enumerate(self.sliceNames):
            self.sliceNames[i] = propertyName + '_' + name
            # self.fileNames[i] = self.sliceNames + fileExt
            # self.fileNames[i] = sliceNames[i]

        self.slicesVal, self.slicesOrientate, self.slicesCoor = {}, {}, {}
        # Go through all specified slices
        # and append coordinates,, slice type (vertical or horizontal), and slice values each to dictionaries
        # Keys are slice names
        for sliceName in self.sliceNames:
            vals = np.genfromtxt(self.caseFullPath + sliceName + sliceNamesSub + fileExt)
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
    @jit
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
    @jit
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
    # @njit
    def getBarycentricMapCoordinates(eigValsGrid):
        # Coordinates of the anisotropy tensor in the tensor basis {a1c, a2c, a3c}. From Banerjee (2007),
        # C1c = lambda1 - lambda2,
        # C2c = 2(lambda2 - lambda3),
        # C3c = 3lambda3 + 1
        c1 = eigValsGrid[:, :, 0] - eigValsGrid[:, :, 1]
        # Not used?
        # c2 = 2*(eigValsGrid[:, :, 1] - eigValsGrid[:, :, 2])
        c3 = 3*eigValsGrid[:, :, 2] + 1
        # Corners of the barycentric triangle
        # Can be random coordinates?
        x1c, x2c, x3c = 1., 0., 1/2.
        y1c, y2c, y3c = 0, 0, np.sqrt(3)/2
        # xBary, yBary = c1*x1c + c2*x2c + c3*x3c, c1*y1c + c2*y2c + c3*y3c
        xBary, yBary = c1 + 0.5*c3, y3c*c3

        # yViolate = yBary - y1c
        return xBary, yBary


    @staticmethod
    @timer
    @jit
    def mergeHorizontalComponents(valsDecomp):
        valsDecomp['hor'] = np.sqrt(valsDecomp['0']**2 + valsDecomp['1']**2)
        return valsDecomp













if __name__ is '__main__':
    import time as t
    time = 'latest'
    caseDir = 'J:'
    caseName = 'ALM_N_H_ParTurb'
    propertyName = 'uuPrime2'
    sliceNames = 'alongWindRotorOne'
    # Orientation of x-axis in x-y plane, in case of angled flow direction
    # Only used for values decomposition
    # Angle in rad and counter-clockwise
    xOrientate = 6/np.pi
    precisionX, precisionY, precisionZ = 1000j, 1000j, 333j
    interpMethod = 'linear'

    case = SliceProperties(time = time, caseDir = caseDir, caseName = caseName, xOrientate = xOrientate)

    case.readSlices(propertyName = propertyName, sliceNames = sliceNames)

    for sliceName in case.sliceNames:
        x2D, y2D, z2D, valsDecomp = \
            case.interpolateDecomposedSliceData(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], case.slicesVal[sliceName], sliceOrientate = case.slicesOrientate[sliceName], xOrientate = case.xOrientate, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)

        # # Another implementation of processAnisotropyTensor() in Cython
        # t0 = t.time()
        # # 682 s
        # valsDecomp, tensors, eigValsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor(valsDecomp)
        # t1 = t.time()
        # ticToc = t1 - t0

        valsDecomp, tensors, eigValsGrid = case.processAnisotropyTensor(valsDecomp)

        xBary, yBary = case.getBarycentricMapCoordinates(eigValsGrid)




        # pickle.dump(tensors, open(case.resultPath + sliceName + '_tensors.p', 'wb'))
        # pickle.dump(x2D, open(case.resultPath + sliceName + '_x2D.p', 'wb'))
        # pickle.dump(y2D, open(case.resultPath + sliceName + '_y2D.p', 'wb'))
        # pickle.dump(z2D, open(case.resultPath + sliceName + '_z2D.p', 'wb'))
        #
        # import RBG_barycentric_colors_clean




        # valsDecomp = case.mergeHorizontalComponents(valsDecomp)
