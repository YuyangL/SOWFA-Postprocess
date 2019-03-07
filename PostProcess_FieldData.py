import os
import numpy as np
from warnings import warn
import Ofpp as of
import random
from numba import njit, jit, prange
from Utilities import timer
from matplotlib import path
from scipy.interpolate import griddata, Rbf

class FieldData:
    def __init__(self, fields = 'all', times = 'all', caseName = 'ABL_N_H', caseDir = './', fileNamePre = '', fileNameSub
    = '', cellCenters = ('ccx', 'ccy', 'ccz'), resultFolder = 'Result'):
        self.fields, self.times = fields, times
        self.caseFullPath = caseDir + '/' + caseName + '/Fields/'
        self.fileNamePre, self.fileNameSub = fileNamePre, fileNameSub
        self.cellCenters = cellCenters
        self.caseTimeFullPaths, self.resultPath = {}, {}
        # If times in list/tuple or all times requested
        if isinstance(times, (list, tuple)) or times in ('all', 'All'):
            # If all times, try remove the result folder from found directories
            if times in ('all', 'All'):
                self.times = os.listdir(self.caseFullPath)
                try:
                    self.times.remove(resultFolder)
                except ValueError:
                    pass

            # Go through all provided times
            for time in self.times:
                self.caseTimeFullPaths[str(time)] = self.caseFullPath + str(time) + '/'
                # In the result directory, there are time directories
                self.resultPath[str(time)] = caseDir + '/' + caseName + '/Fields/' + resultFolder + '/' + str(time) + '/'
                # Try to make the result directories, if not existent already
                try:
                    os.makedirs(self.resultPath[str(time)])
                except OSError:
                    pass

        # Else if only one time provided
        else:
            self.caseTimeFullPaths[str(times)] = self.caseFullPath + str(times) + '/'
            # Make sure times is in list(str)
            self.times = [str(times)]
            self.resultPath[str(times)] = caseDir + '/' + caseName + '/Fields/' + resultFolder + '/' + str(times) + '/'
            try:
                os.makedirs(self.resultPath[str(times)])
            except OSError:
                pass

        # If fields is 'all'/'All'
        # Get a random time directory in order to collect fields
        _, self.caseTimeFullPathsRand = random.choice(list(self.caseTimeFullPaths.items()))
        if fields in ('all', 'All'):
            self.fields = os.listdir(self.caseTimeFullPathsRand)
            # Remove cellCenter fields from this list
            self.fields = [val for _, val in enumerate(self.fields) if val not in cellCenters]
            # Try removing uniform folder is it exists
            try:
                self.fields.remove('uniform')
            except OSError:
                pass

        # Else if provided fields not in a list/tuple
        # Convert str to list
        elif not isinstance(fields, (list, tuple)):
            self.fields = [fields]

        print('\nFieldData object initialized')


    # [DEPRECATED]
    def createSliceData(self, fieldData, baseCoordinate = (0, 0, 90), normalVector = (0, 0, 1)):
        from Utilities import takeClosest
        # Only support horizontal or vertical slices
        # meshSize has to be in the order of x, y, z
        # A plane refers to a horizontal plane
        # cellsPerPlane = self.meshSize[0]*self.meshSize[1]
        # nPlane = self.meshSize[2]

        # Vertical slice, ignore z of baseCoordinate and normalVector
        # x has to be non-negative
        if normalVector[2] == 0:
            # If x = 0, then slice angle is 0 deg and is in xz plane
            if normalVector[0] == 0:
                angleXY = 0.
            else:
                angleXY = 0.5*np.pi + np.arctan(normalVector[1]/normalVector[0])

            # Draw a line across xy-plane that passes baseCoordinate[:2]
            # y = kx + b
            k = np.tan(angleXY)
            b = baseCoordinate[1] - k*baseCoordinate[0]

            ccx, ccy, ccz, cc = self.readCellCenterCoordinates()
            # # Find out cellSize, 0.99 of real cell size to avoid machine error
            # # If smaller than 1, then set it to 1
            # cellSizeEff = max(min(0.99*(ccx[1] - ccx[0]), cellSize), 1)
            meshSize, cellSizeMin, _, ccy3D, _ = self.getMeshInfo(ccx, ccy, ccz)
            lineXs = np.arange(0, max(ccx), cellSizeMin[0])
            lineYs = k*lineXs + b

            # Find y limits
            yLims = (k*ccx[0] + b, k*ccx[meshSize[0] - 1] + b)
            ccyLow, ccyLowIdx, _ = takeClosest(ccy3D[0, :, 0], yLims[0])
            ccyHi, ccyHiIdx, _ = takeClosest(ccy3D[0, :, 0], yLims[1])

            # Find target x for each y coordinate of the cells in a horizontal plane
            xTargets = []
            i = ccyLowIdx
            while i <= ccyHiIdx:
                xTargets.append((ccy3D[0, i, 0] - b)/k)
                i += 1

            # For the lowest horizontal plane, find all x coordinates and cell indices of this vertical slice
            # cellCoorXs = []
            iCells = []
            # The start row of x, i.e. starting y index is ccyLowIdx
            rowX = ccyLowIdx
            for xTarget in xTargets:
                val, iCellX, diff = takeClosest(ccx[:meshSize[0]], xTarget)
                # cellCoorXs.append(cellCoorX)
                # Every next find is on the next row of x, i.e. next y
                # Position in a row, iCellX, + row shift, rowX*meshSize[0]
                iCells.append(iCellX + rowX*meshSize[0])

                rowX += 1

            iCells = np.array(iCells)
            # Repeat for all horizontal planes
            iPlane = 1
            # Lowest plane of vertical slice cell indices
            iCellsBase = iCells
            while iPlane < meshSize[2]:
                # Next plane of vertical slice cell indices are current plane's + a whole plane
                iCellsNext = iCellsBase + iPlane*meshSize[0]*meshSize[1]
                iCells = np.hstack((iCells, iCellsNext))
                iPlane += 1

        fieldDataSlice = np.take(fieldData, indices = iCells, axis = 0)
        ccSlice = np.take(cc, indices = iCells, axis = 0)

        # Dimension of the slice, (size in diagonal, size in vertical)
        sliceDim = (len(iCellsBase), 1, iPlane)

        print('\nField data slice created')
        return fieldDataSlice, ccSlice, sliceDim


    @timer
    @jit(parallel = True)
    def readFieldData(self, time = None):
        fieldData = {}
        for field in self.fields:
            # if time is None:
            #     fieldData[field] = of.parse_internal_field(self.caseTimeFullPaths[self.times[0]] + field)
            # else:
            #     fieldData[field] = of.parse_internal_field(self.caseTimeFullPaths[str(time)] + field)

            # Read the data of field in the 1st time directory
            fieldData[field] = of.parse_internal_field(self.caseTimeFullPaths[self.times[0]] + field)
            # If multiple times requested, read data and stack them in 3rd D
            if len(self.times) > 1:
                for i in prange(1, len(self.times)):
                    print(i)
                    fieldData_i = of.parse_internal_field(self.caseTimeFullPaths[self.times[i]] + field)
                    fieldData[field] = np.dstack((fieldData[field], fieldData_i))

        print('\n' + str(self.fields) + ' data read, if multiple times requested, data of different times are stacked in 3D')
        return fieldData


    def readCellCenterCoordinates(self):
        try:
            # cellCenters has to be in the order of x, y, z
            ccx = of.parse_internal_field(self.caseTimeFullPaths[self.times[-1]] + self.cellCenters[0])
            ccy = of.parse_internal_field(self.caseTimeFullPaths[self.times[-1]] + self.cellCenters[1])
            ccz = of.parse_internal_field(self.caseTimeFullPaths[self.times[-1]] + self.cellCenters[2])
            cc = np.vstack((ccx, ccy, ccz))
        except:
            warn('\nCell centers have to be stored in the latest time directory', stacklevel = 2)

        print('\nCell center coordinates read')
        return ccx, ccy, ccz, cc.T


    @staticmethod
    @timer
    @njit(parallel = True)
    def confineFieldDomain(x, y, z, vals, bndX = (None, None), bndY = (None, None), bndZ = (None, None), planarRot = 0):
        # assert isinstance(bndX, (list, tuple))
        # assert isinstance(bndY, (list, tuple))
        # assert isinstance(bndZ, (list, tuple))
        # Change all None to either min or max values of the domain
        for i in prange(2):
            # For the lower bound
            if i == 0:
                bndX[i] = np.min(x) if bndX[i] is None else bndX[i]
                bndY[i] = np.min(y) if bndY[i] is None else bndY[i]
                bndZ[i] = np.min(z) if bndZ[i] is None else bndZ[i]
            # For the upper bound
            else:
                bndX[i] = np.max(x) if bndX[i] is None else bndX[i]
                bndY[i] = np.max(y) if bndY[i] is None else bndY[i]
                bndZ[i] = np.max(z) if bndZ[i] is None else bndZ[i]

        # for xVal, yVal, zVal, val in zip(x, y, z, vals):
        xNew, yNew, zNew, valsNew = [], [], [], []
        for i in prange(len(x)):
            if x[i] >= bndX[0] and x[i] <= bndX[1] \
                and y[i] >= bndY[0] and y[i] <= bndY[1] \
                and z[i] >= bndZ[0] and z[i] <= bndZ[1]:
                xNew.append(x[i])
                yNew.append(y[i])
                zNew.append(z[i])
                valsNew.append(vals[i])

        return xNew, yNew, zNew, valsNew


    @staticmethod
    @timer
    @jit(parallel = True, fastmath = True)
    def confineFieldDomain_Rotated(x, y, z, vals, boxL, boxW, boxH, boxO = (0, 0, 0), boxRot = 0):
        print('\nConfining field domain with rotated box...')
        # Create the bounding box
        box = path.Path(((boxO[0], boxO[1]),
                         (boxL*np.cos(boxRot) + boxO[0], boxL*np.sin(boxRot) + boxO[1]),
                         (boxL*np.cos(boxRot) + boxO[0] - boxW*np.sin(boxRot), boxL*np.sin(boxRot) + boxO[1] + boxW*np.cos(boxRot)),
                         (boxO[0] - boxW*np.sin(boxRot), boxO[1] + boxW*np.cos(boxRot)),
                         (boxO[0], boxO[1])))
        # First confine within z range
        xNew, yNew, zNew, valsNew = [], [], [], []
        cnt, milestone = 0, 25
        for i in prange(len(x)):
            if z[i] < boxH + boxO[2] and z[i] > boxO[2]:
                xNew.append(x[i])
                yNew.append(y[i])
                zNew.append(z[i])
                valsNew.append(vals[i])

        # Then confine within x and y range
        xy = (np.vstack((xNew, yNew))).T
        # Bool flags
        flags = box.contains_points(xy)
        xNew2, yNew2, zNew2, valsNew2 = [], [], [], []
        for i in prange(len(flags)):
            if flags[i]:
                xNew2.append(xNew[i])
                yNew2.append(yNew[i])
                zNew2.append(zNew[i])
                valsNew2.append(valsNew[i])

            # Gauge progress
            cnt += 1
            progress = cnt/(len(flags) + 1)*100.
            if progress >= milestone:
                print(' ' + str(milestone) + '%...', end = '')
                milestone += 25

        return np.array(xNew2), np.array(yNew2), np.array(zNew2), np.array(valsNew2), box, flags


    @staticmethod
    @timer
    # @jit(parallel = True, fastmath = True)
    def interpolateFieldData(x, y, z, vals, precisionX = 1500j, precisionY = 1500j, precisionZ = 500j, interpMethod = 'linear'):
        print('\nInterpolating field data...')
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1.000001, 0.999999)
        knownPts = np.vstack((x, y, z)).T
        # Interpolate x, y, z to designated precisions
        x3D, y3D, z3D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX,
                        y.min()*bnd[0]:y.max()*bnd[1]:precisionY,
                        z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
        # requestPts = np.vstack((x3D.ravel(), y3D.ravel(), z3D.ravel())).T
        requestPts = (x3D.ravel(), y3D.ravel(), z3D.ravel())
        dim = vals.shape
        orders, milestone, cnt = [], 2, 0
        # If vals is 1D, i.e. scalar field
        if len(dim) == 1:
            print('\nInterpolating scalar field...')
            valsND = griddata(knownPts, vals, requestPts, method = interpMethod)
        # If vals is 2D
        elif len(dim) == 2:
            # Initialize nRow x nCol x nComponent array valsND by interpolating 1st component of the values, x, or xx,
            # as others come later in the loop below
            valsND = griddata(knownPts, vals[:, 0].ravel(), requestPts, method = interpMethod)
            print('good')
            # Then go through the rest components and stack them in 4D
            for i in prange(1, vals.shape[1]):
                print(i)
                # Add i to a list in case prange is not ordered
                orders.append[i]
                # For one component
                valsND_i = griddata(knownPts, vals[:, i].ravel(), requestPts, method = interpMethod)
                # Then stack them as the 4th D
                valsND = np.dstack((valsND, valsND_i))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[1] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]
        # If vals is 3D or more
        else:
            print('good')
            # If the vals is 4D or above, reduce it to 3D
            if len(dim) > 3:
                print('better')
                vals = np.reshape(vals, (vals.shape[0], vals.shape[1], -1))
                print(vals.shape)

            # Initialize nRow x nCol x nComponent array valsND by interpolating 1st component of the values, x, or xx,
            # as others come later in the loop below
            valsND = griddata(knownPts, vals[:, :, 0].ravel(), requestPts, method = interpMethod)
            # Then go through the rest components and stack them in 4D
            for i in prange(1, vals.shape[2]):
                print(i)
                # Add i to a list in case prange is not ordered
                orders.append[i]
                # For one component
                valsND_i = griddata(knownPts, vals[:, :, i].ravel(), requestPts, method = interpMethod)
                # Then stack them as the 4th D
                valsND = np.dstack((valsND, valsND_i))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[2] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]

        # If vals was 4D, then valsND should be 5D
        if len(dim) == 4:
            valsND = np.reshape(valsND, (valsND.shape[0], valsND.shape[1], dim[2], dim[3]))

        return x3D, y3D, z3D, valsND


    @staticmethod
    @timer
    @jit(parallel = True, fastmath = True)
    def interpolateFieldData_RBF(x, y, z, vals, precisionX = 1500j, precisionY = 1500j, precisionZ = 500j,
                             function = 'linear'):
        print('\nInterpolating field data...')
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1.000001, 0.999999)
        # knownPts = np.vstack((x, y, z)).T
        # Interpolate x, y, z to designated precisions
        x3D, y3D, z3D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX,
                        y.min()*bnd[0]:y.max()*bnd[1]:precisionY,
                        z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
        # requestPts = np.vstack((x3D.ravel(), y3D.ravel(), z3D.ravel())).T
        dim = vals.shape
        orders, milestone, cnt = [], 2, 0
        if len(dim) == 1:
            rbf = Rbf(x, y, z, vals, function = function)
            valsND = rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())
        elif len(dim) == 2:
            rbf = Rbf(x, y, z, vals[:, 0], function = function)
            valsND = rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())
            for i in prange(1, dim[1]):
                print(i)
                orders.append(i)
                rbf = Rbf(x, y, z, vals[:, i], function = function)
                valsND = np.vstack((valsND, rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[2] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]
        else:
            # If the vals is 4D or above, reduce it to 3D
            if len(dim) > 3:
                vals = np.reshape(vals, (vals.shape[0], vals.shape[1], -1))

            rbf = Rbf(x, y, z, vals[:, :, 0], function = function)
            valsND = rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())
            for i in prange(1, dim[1]):
                print(i)
                orders.append(i)
                rbf = Rbf(x, y, z, vals[:, :, i], function = function)
                valsND = np.vstack((valsND, rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[2] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]

        return x3D, y3D, z3D, valsND






    def getMeshInfo(self, ccx, ccy, ccz):
        from Utilities import getArrayStepping
        # Mesh size in x
        valOld = ccx[0]
        for i, val in enumerate(ccx[1:]):
            if val < valOld:
                meshSizeX = i + 1
                break

            valOld = val
            # meshSizeX = i + 2

        # Mesh size in y
        i = meshSizeX
        valOld = ccy[0]
        count = 1
        while i < ccy.size:
            if ccy[i] < valOld:
                meshSizeY = count
                break

            valOld = ccy[i]
            i += meshSizeX
            # meshSizeY = count
            count += 1

        # meshSizeY = int((np.argmax(ccy) + 1)/meshSizeX)
        # meshSizeY = np.count_nonzero(ccz == min(ccz))/meshSizeX
        meshSize = (meshSizeX, meshSizeY, int(ccz.size/(meshSizeX*meshSizeY)))

        # 1D to 3D array, access by [iX, iY, iZ]
        # List of x values: ccx3D[:, 0, 0]
        # No clue, don't ask why
        ccx3D = ccx.reshape((meshSize[2], meshSize[1], meshSize[0])).T
        ccy3D = ccy.reshape((meshSize[2], meshSize[1], meshSize[0])).T
        ccz3D = ccz.reshape((meshSize[2], meshSize[1], meshSize[0])).T

        diffMinX, diffMinY, diffMinZ = \
            getArrayStepping(ccx3D[:, 0, 0]), \
            getArrayStepping(ccy3D[0, :, 0]), \
            getArrayStepping(ccz3D[0, 0, :])
        cellSizeMin = (diffMinX, diffMinY, diffMinZ)

        print('\nMesh info retrieved')
        return meshSize, cellSizeMin, ccx3D, ccy3D, ccz3D


    def convertScalarFieldToMeshGrid(self, scalarField, meshSize = None):
        scalarField = np.array(scalarField)
        if len(scalarField.shape) > 2:
            warn('\nProvided scalarField can only have two dimensions or less! Not converting to mesh grid!\n', 
                 stacklevel = 2)
            return scalarField
        
        # If meshSize is not provided, infer
        if meshSize is None:
            ccx, ccy, ccz, _ = self.readCellCenterCoordinates()
            meshSize, _, _, _, _ = self.getMeshInfo(ccx, ccy, ccz)

        # scalarField3D is has the index of [x, y, z]
        scalarField3D = scalarField.reshape((meshSize[2], meshSize[1], meshSize[0])).T
        print('\nConverted scalar field to mesh grid')
        return scalarField3D


    def createTemporalMeanFields(self):
        if isinstance(self.times, (float, int)):
            warn('\nOnly one time provided, no temporal mean is calculated!\n', stacklevel = 2)
            fieldData = self.readFieldData()
            return fieldData

        fieldDataMean = {}
        for field in self.fields:
            iTime = 0
            # First mean is first time itself
            fieldDataMean[field] = of.parse_internal_field(self.caseTimeFullPaths[self.times[iTime]] + field)
            while iTime < (len(self.times) - 1):
                fieldDataTime = of.parse_internal_field(self.caseTimeFullPaths[self.times[iTime + 1]] + field)
                fieldDataMean[field] += fieldDataTime
                iTime += 1

            fieldDataMean[field] /= len(self.times)

        print('\nMean field data created')
        return fieldDataMean


    def decomposedFields(self, vectorField2D):
        # Decompose the vector field into x, y, z, and xy components
        vectorField2D = np.array(vectorField2D)
        if len(vectorField2D.shape) != 2:
            warn('\nThe shape of the vectorField has to be two dimensions. Do not provide mesh grid data! No '
                 'decomposition is done!\n',
                 stacklevel = 2)
            return vectorField2D

        if vectorField2D.shape[0] == 3:
            vectorField2D = vectorField2D.T

        fieldX2D, fieldY2D, fieldZ2D, fieldHor2D = \
            np.zeros((vectorField2D.shape[0], 1)), \
            np.zeros((vectorField2D.shape[0], 1)), \
            np.zeros((vectorField2D.shape[0], 1)), \
            np.zeros((vectorField2D.shape[0], 1))
        for i, row in enumerate(vectorField2D):
            fieldX2D[i], fieldY2D[i], fieldZ2D[i] = row[0], row[1], row[2]
            fieldHor2D[i] = np.sqrt(row[0]**2 + row[1]**2)

        print('\nDecomposed vector field to horizontal and vertical components')
        return fieldX2D, fieldY2D, fieldZ2D, fieldHor2D
    

    def getPlanerFluctuations(self, fieldHor3D, fieldZ3D):
        fieldHor3D, fieldZ3D = np.array(fieldHor3D), np.array(fieldZ3D)
        if (len(fieldHor3D.shape) != 3) or (len(fieldZ3D.shape) != 3):
            warn('\nfieldHor3D and fieldZ3D have to have three indices in the order of [iX, iY, iZ]! Not creating '
                 'spatial '
                 'mean '
                 'fields!\n', stacklevel = 2)
            return fieldHor3D, fieldZ3D
        
        fieldHorMean, fieldZmean = \
            np.zeros((fieldHor3D.shape[2], 1)), np.zeros((fieldZ3D.shape[2], 1))
        fieldHorRes3D, fieldZres3D = fieldHor3D, fieldZ3D
        for i in range(fieldHor3D.shape[2]):
            fieldHorMean[i] = fieldHor3D[:, :, i].mean()
            fieldZmean[i] = fieldZ3D[:, :, i].mean()
            fieldHorRes3D[:, :, i] -= fieldHorMean[i]
            fieldZres3D[:, :, i] -= fieldZmean[i]

        print('\nFinished planer fluctuation fields calculation')
        return fieldHorRes3D, fieldZres3D, fieldHorMean, fieldZmean
            




        
        
        





            
        
