import os
import numpy as np
from warnings import warn
import Ofpp as of
import random

class FieldData(object):
    def __init__(self, fields = 'all', times = 'all', caseName = 'ABL_N_H/Field', caseDir = './', fileNamePre = '', fileNameSub
    = '', cellCenters = ('ccx', 'ccy', 'ccz'), cellSizeMin = None, meshSize = None):
        self.fields, self.times = fields, times
        self.caseFullPath = caseDir + '/' + caseName + '/'
        self.fileNamePre, self.fileNameSub = fileNamePre, fileNameSub
        self.cellCenters = cellCenters

        self.caseTimeFullPaths = {}
        # If times in list/tuple, then get the average
        if isinstance(times, (list, tuple)):
            for time in times:
                self.caseTimeFullPaths[str(time)] = self.caseFullPath + str(time) + '/'
        elif times in ('all', 'All'):
            self.times = os.listdir(self.caseFullPath)
            try:
                self.times.remove('Result')
            except:
                pass

            for time in self.times:
                self.caseTimeFullPaths[str(time)] = self.caseFullPath + str(time) + '/'
        else:
            self.caseTimeFullPaths[str(times)] = self.caseFullPath + str(times) + '/'
            # Make sure times is in list(str)
            self.times = [str(times)]

        # If fields is 'all'/'All'
        # Get a random time directory in order to collect fields
        _, self.caseTimeFullPathsRand = random.choice(list(self.caseTimeFullPaths.items()))
        if fields in ('all', 'All'):
            self.fields = os.listdir(self.caseTimeFullPathsRand)
            # Remove cellCenter fields from this list
            self.fields = [val for i, val in enumerate(self.fields) if val not in cellCenters]
        # Convert str to list
        elif not isinstance(fields, (list, tuple)):
            self.fields = [fields]

        print('\nFieldData object initialized')


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


    def readFieldData(self, time = None):
        fieldData = {}
        for field in self.fields:
            if time is None:
                fieldData[field] = of.parse_internal_field(self.caseTimeFullPaths[self.times[0]] + field)
            else:
                fieldData[field] = of.parse_internal_field(self.caseTimeFullPaths[str(time)] + field)

            print('\nField(s) data read')
        return fieldData


    def readCellCenterCoordinates(self):
        # cellCenters has to be in the order of x, y, z
        ccx = of.parse_internal_field(self.caseTimeFullPathsRand + self.cellCenters[0])
        ccy = of.parse_internal_field(self.caseTimeFullPathsRand + self.cellCenters[1])
        ccz = of.parse_internal_field(self.caseTimeFullPathsRand + self.cellCenters[2])
        cc = np.vstack((ccx, ccy, ccz))

        print('\nCell center coordinates read')
        return ccx, ccy, ccz, cc.T


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
            




        
        
        





            
        
