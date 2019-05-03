import os
import numpy as np
from warnings import warn
from Utilities import timer
from numba import njit, jit, prange
import shutil

class BaseProperties:
    def __init__(self, caseName, caseDir = '.', filePre = '', fileSub = '', ensembleFolderName = 'Ensemble', resultFolder = 'Result', timeCols = 'infer', timeKw = 'ime', forceRemerge = False, **kwargs):
        self.caseName, self.caseDir = caseName, caseDir
        self.caseFullPath = caseDir + '/' + caseName + '/'
        # self.startTime, self.stopTime = startTime, stopTime
        self.filePre, self.fileSub = filePre, fileSub
        # Check if result folder is made
        self.resultDir = self.caseFullPath + resultFolder + '/'
        try:
            os.makedirs(self.resultDir)
        except OSError:
            pass

        self.ensembleFolderPath = self.caseFullPath + ensembleFolderName + '/'
        self.timeCols = (timeCols,) if isinstance(timeCols, int) else timeCols
        self._mergeTimeDirectories(timeKw = timeKw, forceRemerge = forceRemerge, **kwargs)
        self.timesAll = self._readTimes(**kwargs)
        # self.timesSelected, self.startTimeReal, self.stopTimeReal, self.iStart, self.iStop = self._selectTimes(startTime = startTime, stopTime = stopTime)
        self.propertyData, self.fileNames = {}, []

        print('\n{0} initialized'.format(caseName))


    def _ensureTupleInput(self, input):
        inputTuple = (input,) if isinstance(input, (str, np.ndarray, int)) else input
        # If input[0] is '*' or 'all', get all file names
        # inputTuple = os.listdir(self.ensembleFolderPath) if inputTuple[0] in ('*', 'all') else inputTuple
        if inputTuple[0] in ('*', 'all'):
            # Ignore hidden files
            inputTuple = tuple([f for f in os.listdir(self.ensembleFolderPath) if not f.startswith('.')])

        return inputTuple


    def _readTimes(self, noRepeat = True, **kwargs):
        fileNames = os.listdir(self.ensembleFolderPath)
        # In case of file e.g. hLevelsCell that doesn't incl. times
        try:
            timesAll = np.genfromtxt(self.ensembleFolderPath + '/' + fileNames[0])[:, self.timeCols[0]]
        except IndexError:
            timesAll = np.genfromtxt(self.ensembleFolderPath + '/' + fileNames[1])[:, self.timeCols[0]]
        # # In case of inflow profile
        # try:
        #     timesAll = np.genfromtxt(self.ensembleFolderPath + self.filePre + 'U_mean' + self.fileSub)[:, self.timeCols]
        # # In case of turbineOutput
        # except IOError:
        #     timesAll = np.genfromtxt(self.ensembleFolderPath + self.filePre + 'Cl' + self.fileSub)[:, self.timeCols]

        if noRepeat:
            timesAll = np.unique(timesAll)

        return timesAll


    @jit(parallel = True)
    def _mergeTimeDirectories(self, trimOverlapTime = True, timeKw = 'ime', forceRemerge = False, excludeFile = None):
        # [DEPRECATED]
        # @jit
        # def takeClosestIdx(lists, vals):
        #     (lists, vals) = (iter(lists), iter(vals)) if not isinstance(lists, Iterator) else (lists, vals)
        #     idxs = []
        #     while any(True for _ in lists):
        #         idx = np.searchsorted(next(lists), next(vals))
        #         idxs.append(idx)
        #
        #     return idxs

        def numberStringToFloat(str):
            return float(str)

        # Check whether the directory is made
        try:
            os.makedirs(self.ensembleFolderPath)
        except OSError:
            # If folder not empty, abort
            if os.listdir(self.ensembleFolderPath) and not forceRemerge:
                print('\n{0} files already exist'.format(self.ensembleFolderPath))
                return

        # Available time directories (excluding Ensemble and Result) and file names
        timeDirs = os.listdir(self.caseFullPath)[:-2]
        # Sort the time directories
        timeDirs.sort(key = numberStringToFloat)
        fileNames = os.listdir(self.caseFullPath + timeDirs[0])
        # In case excludeFile is provided, remove it from fileNames
        if excludeFile is not None:
            excludeFile = self._ensureTupleInput(excludeFile)
            for i in prange(len(excludeFile)):
                try:
                    fileNames.remove(excludeFile[i])
                except ValueError:
                    warn('\n' + self.caseName + ' does not have ' + excludeFile[i] + ' to exclude!', stacklevel = 2)
                    pass

        # Initialize ensemble files
        fileEnsembles = {}
        for i in prange(len(fileNames)):
            fileEnsembles[fileNames[i]] = open(self.ensembleFolderPath + fileNames[i], "w")
        # for fileName in fileNames:
        #     fileEnsembles[fileName] = open(self.ensembleFolderPath + fileName, "w")

        # self.timeCols = (self.timeCols,) if isinstance(self.timeCols, int) else self.timeCols

        if self.timeCols == 'infer':
            self.timeCols = []
            for fileName in fileNames:
                with open(self.caseFullPath + timeDirs[0] + '/' + fileName, 'r') as file:
                    header = (file.readline()).split()
                    self.timeCols.append(header.index(list(filter(lambda kw: timeKw in kw, header))[0]))
        else:
            self.timeCols *= len(fileNames)

        # Go through time folders and append files to ensemble
        # Excluding Ensemble folder
        for i in range(len(timeDirs)):
            # If trim overlapped time and not in last time directory
            if trimOverlapTime and i < len(timeDirs) - 1:
                knownTimeCols, times, iTrim = {}, {}, {}
                # Go through all time columns of each file in order
                for j in range(len(self.timeCols)):
                    # Retrieve list of time and trim index information for jth file in ith time directory
                    # After each retrieval, add this time column to known time column dictionary as key
                    # and corresponding file name as value
                    if str(self.timeCols[j]) not in knownTimeCols.keys():
                        try:
                            times[fileNames[j]] = np.genfromtxt(self.caseFullPath + timeDirs[i] + '/' + fileNames[j])[:, self.timeCols[j]]
                        # In case the last line wasn't written properly,
                        # which means the simulation was probably aborted, discard the last line
                        except ValueError:
                            times[fileNames[j]] = np.genfromtxt(self.caseFullPath + timeDirs[i] + '/' + fileNames[j], skip_footer = 1)[:, self.timeCols[j]]

                        # Index at which trim should start for this file
                        iTrim[fileNames[j]] = np.searchsorted(times[fileNames[j]], np.float_(timeDirs[i + 1]))
                        # Add this time column to known time column list
                        knownTimeCols[str(self.timeCols[j])] = fileNames[j]

                    # If current time column already exists in remembered dictionary,
                    # then skip it and retrieve the file name the last time it had this number of time column
                    else:
                        times[fileNames[j]] = times[knownTimeCols[str(self.timeCols[j])]]
                        iTrim[fileNames[j]] = iTrim[knownTimeCols[str(self.timeCols[j])]]

            # Go through each file in this time directory
            for fileName in fileNames:
                # If trim overlapped time and not last time directory and trim is indeed needed
                # print(i, len(timeDirs) - 1)
                # print(iTrim[fileName], len(times[fileName]) - 1)
                if trimOverlapTime and i < len(timeDirs) - 1 and iTrim[fileName] < (len(times[fileName]) - 1):
                    with open(self.caseFullPath + timeDirs[i] + '/' + fileName, 'r') as file:
                        # Filter out empty lines before iTrim indices can be mapped
                        lines = list(filter(None, (line.rstrip() for line in file)))

                    # for line in lines:
                    #     lines = filter(None, )

                    # print(f'\nTrimming overlapped time and adding {fileName} from {timeDirs[i]} to Ensemble...')
                    print('\nTrimming overlapped time and adding {0} from {1} to Ensemble...'.format(
                            fileName, timeDirs[i]))
                    # Writelines support writing a 1D list, since lines is 2D,
                    # join each row with "\n"
                    # Note: the header of 2nd file onward will still be written in ensemble,
                    # just that when reading file into array using numpy, the headers should automatically be ignored
                    # since it starts with "#"
                    # Write the 1st line as empty new line so that the 1st line of lines is not on the same line as last line of fileEnsembles
                    fileEnsembles[fileName].writelines("\n")
                    fileEnsembles[fileName].writelines("\n".join(lines[:iTrim[fileName] + 1]))
                # Otherwise, append this file directly to Ensemble
                else:
                    # print(f'\nAdding {fileName} from {timeDirs[i]} to Ensemble...')
                    print('\nAdding {0} from {1} to Ensemble...'.format(fileName, timeDirs[i]))
                    # Again, write the 1st line as empty new line to avoid 1st line of next file being on the same line of old file
                    fileEnsembles[fileName].writelines("\n")
                    fileEnsembles[fileName].write(open(self.caseFullPath + timeDirs[i] + '/' + fileName).read())

        print("\nMerged time directories for " + str(self.caseName) + " files are stored at:\n " + str(self.ensembleFolderPath))


    def _selectTimes(self, startTime = None, stopTime = None):
        startTime = self.timesAll[0] if startTime is None else startTime
        stopTime = self.timesAll[len(self.timesAll)] if stopTime is None else stopTime
        # Bisection left to find actual starting and ending time and their indices
        (iStart, iStop) = np.searchsorted(self.timesAll, (startTime, stopTime))
        # If stopTime larger than any time, iStop = len(timesAll)
        iStop = min(iStop, len(self.timesAll) - 1)
        startTimeReal, stopTimeReal = self.timesAll[iStart], self.timesAll[iStop]

        timesSelected = self.timesAll[iStart:iStop]

        # print('\nTime and index information extracted for ' + str(startTimeReal) + ' s - ' + str(stopTimeReal) + ' s')
        return timesSelected, startTimeReal, stopTimeReal, iStart, iStop


    @jit(parallel = True)
    def readPropertyData(self, fileNames = ('*',), skipRow = 0, skipCol = 0, skipFooter = 0):
        self.fileNames = self._ensureTupleInput(fileNames)
        skipRow = (skipRow,)*len(self.fileNames) if isinstance(skipRow, int) else skipRow
        skipCol = (skipCol,)*len(self.fileNames) if isinstance(skipCol, int) else skipCol
        skipFooter = (skipFooter,)*len(self.fileNames) if isinstance(skipFooter, int) else skipFooter

        for i in prange(len(self.fileNames)):
            # Data dictionary of specified property(s) of all times
            self.propertyData[self.fileNames[i]] = \
                np.genfromtxt(self.ensembleFolderPath + self.filePre + self.fileNames[i] + self.fileSub, skip_footer = skipFooter[i])[skipRow[i]:,
                skipCol[i]:]
        # for fileName in self.fileNames:
        #     # Data dictionary of specified property(s) of all times
        #     self.propertyData[fileName] = \
        #         np.genfromtxt(self.ensembleFolderPath + self.filePre + fileName + self.fileSub)[next(skipRow):, next(skipCol):]

        print('\n' + str(self.fileNames) + ' read')


    # prange has to run with parallel = True
    @jit(parallel = True, fastmath = True)
    def calculatePropertyMean(self, axis = 1, startTime = None, stopTime = None):
        self.timesSelected, _, _, iStart, iStop = self._selectTimes(startTime = startTime, stopTime = stopTime)
        for i in prange(len(self.fileNames)):
            self.propertyData[self.fileNames[i] + '_mean'] = np.mean(self.propertyData[self.fileNames[i]][
                                                                     iStart:iStop],
                                                                  axis = axis)
        # for fileName in self.fileNames:
        #     self.propertyData[fileName + '_mean'] = np.mean(self.propertyData[fileName][iStart:iStop], axis = axis)

        # print(f'\nTemporal average calculated for {self.fileNames} from {self.timesSelected[0]} s - {self.timesSelected[-1]} s')
        print('\nTemporal average calculated for {} from {:.4f} s - {:.4f} s'.format(self.fileNames,
                                                                                   self.timesSelected[0], self.timesSelected[-1]))


    def trimInvalidCharacters(self, fileNames, invalidChars):
        fileNames = self._ensureTupleInput(fileNames)
        invalidChars = (invalidChars,) if isinstance(invalidChars, str) else invalidChars

        for fileName in fileNames:
            with open(self.ensembleFolderPath + fileName, 'r') as f:
                lst = [line.rstrip('\n \t') for line in f]

            for invalidChar in invalidChars:
                lst = [string.replace(invalidChar, '') for string in lst]

            with open(self.ensembleFolderPath + fileName, "w") as f:
                f.writelines('\n'.join(lst))



class BoundaryLayerProfiles(BaseProperties):
    def __init__(self, caseName, fileNameH = 'hLevelsCell', blFolder = 'ABL', **kwargs):
        self.fileNameH = fileNameH
        super(BoundaryLayerProfiles, self).__init__(caseName = caseName + '/' + blFolder, timeCols = 0, excludeFile =
        fileNameH, **kwargs)
        # Copy fileNameH to Ensemble in order to use it later
        time = os.listdir(self.caseFullPath)[0]
        shutil.copy2(self.caseFullPath + time + '/' + fileNameH, self.ensembleFolderPath)


    def readPropertyData(self, fileNames = ('*'), **kwargs):
        # Read height levels
        self.hLvls = np.genfromtxt(self.ensembleFolderPath + self.fileNameH)
        # Override skipCol to suit inflow property files
        # Columns to skip are 0: time; 1: time step
        super(BoundaryLayerProfiles, self).readPropertyData(fileNames = fileNames, skipCol = 2)


    def calculatePropertyMean(self, startTime = None, stopTime = None, **kwargs):
        # Override axis to suit inflow property files
        super(BoundaryLayerProfiles, self).calculatePropertyMean(axis = 0, startTime = startTime, stopTime = stopTime, **kwargs)


class TurbineOutputs(BaseProperties):
    def __init__(self, caseName, dataFolder = 'turbineOutput', globalQuantities = ('powerRotor', 'rotSpeed', 'thrust',
                                                                      'torqueRotor',
                                                       'torqueGen', 'azimuth', 'nacYaw', 'pitch'), **kwargs):
        self.globalQuantities = globalQuantities
        super(TurbineOutputs, self).__init__(caseName + '/' + dataFolder, **kwargs)

        self.nTurb, self.nBlade = 0, 0


    @timer
    @jit(parallel = True, fastmath = True)
    def readPropertyData(self, fileNames = ('*',), skipRow = 0, skipCol = 'infer', verbose = True, turbInfo = ('infer',)):
        fileNames = self._ensureTupleInput(fileNames)
        globalQuantities = (
        'powerRotor', 'rotSpeed', 'thrust', 'torqueRotor', 'torqueGen', 'azimuth', 'nacYaw', 'pitch', 'powerGenerator')
        if skipCol is 'infer':
            skipCol = []
            for file in fileNames:
                if file in globalQuantities:
                    skipCol.append(3)
                else:
                    skipCol.append(4)

        super(TurbineOutputs, self).readPropertyData(fileNames = fileNames, skipRow = skipRow, skipCol = skipCol)

        if turbInfo[0] is 'infer':
            turbInfo = np.genfromtxt(self.ensembleFolderPath + self.filePre + 'Cl' + self.fileSub)[skipRow:, :2]

        # Number of turbines and blades
        (self.nTurb, self.nBlade) = (int(np.max(turbInfo[:, 0]) + 1), int(np.max(turbInfo[:, 1]) + 1))

        fileNamesOld, self.fileNames = self.fileNames, list(self.fileNames)
        for fileName in fileNamesOld:
            for i in range(self.nTurb):
                if fileName not in globalQuantities:
                    for j in range(self.nBlade):
                        newFileName = fileName + '_Turb' + str(i) + '_Bld' + str(j)
                        self.propertyData[newFileName] = self.propertyData[fileName][(i*self.nBlade + j)::(self.nTurb*self.nBlade), :]
                        self.fileNames.append(newFileName)

                else:
                    newFileName = fileName + '_Turb' + str(i)
                    self.propertyData[newFileName] = self.propertyData[fileName][i::self.nTurb]
                    self.fileNames.append(newFileName)

        if verbose:
            print('\n' + str(self.fileNames) + ' read')



class InflowBoundaryField(BaseProperties):
    def __init__(self, caseName, caseDir = '.', boundaryDataFolder = 'boundaryData', avgFolder = 'Average', **kwargs):
        self.caseName, self.caseDir = caseName, caseDir
        self.caseFullPath = caseDir + '/' + caseName + '/' + boundaryDataFolder + '/'
        self.inflowPatches = os.listdir(self.caseFullPath)
        # Try remove "Average" folder from collected patch names
        try:
            self.inflowPatches.remove(avgFolder)
        except ValueError:
            pass

        self.avgFolderPath = self.caseFullPath + avgFolder + '/'
        # Patch folder paths in Average folder
        self.avgFolderPatchPaths, self.casePatchFullPaths = [], []
        for patch in self.inflowPatches:
            self.avgFolderPatchPaths.append(self.avgFolderPath + patch + '/')
            self.casePatchFullPaths.append(self.caseFullPath + patch + '/')
            # Try making Average folder and its subfolder, if not already
            try:
                os.makedirs(self.avgFolderPath + patch + '/')
            except OSError:
                pass

        self.propertyData, self.propertyDataMean = {}, {}
        # Exception for inheritance class DrivingPressureGradient
        try:
            self.timesAll, self.timesAllRaw = self._readTimes(**kwargs)
        except NotADirectoryError:
            pass

        print('{} InflowBoundaryField object initialized'.format(caseName))


    def _readTimes(self, remove = 'points', **kwargs):
        timesAll = os.listdir(self.casePatchFullPaths[0])
        try:
            timesAll.remove(remove)
        except ValueError:
            pass

        # Raw all times that are string and can be integer and float mixed
        # Useful for locating time directories that can be integer
        timesAllRaw = timesAll
        # Numerical float all times and sort from low to high
        timesAll = np.array([float(i) for i in timesAll])
        # Sort string all times by its float counterpart
        timesAllRaw = [timeRaw for time, timeRaw in sorted(zip(timesAll, timesAllRaw))]
        # Use Numpy sort() to sort float all times
        timesAll.sort()

        return timesAll, timesAllRaw


    @timer
    # Parallel doesn't work here due to decode?
    def readPropertyData(self, fileNames = ('*',), skipRow = 22, skipFooter = 1, nTimeSample = -1, lstrPrecision = 12, rstrPrecision = 20):
        def __trimBracketCharacters(data):
            # Get left and right column of U
            dataCol0, dataCol1, dataCol2 = data['f0'], data['f1'], data['f2']
            # New corrected data
            dataNew = np.empty((data.shape[0], 3, data.shape[2]))
            # Go through each point then each time
            for l in range(data.shape[0]):
                # print(l)
                for m in range(data.shape[2]):
                    newVal0, newVal2 = dataCol0[l, 0, m].decode('utf-8'), dataCol2[l, 0, m].decode('utf-8')
                    dataNew[l, 0, m] = float(newVal0.lstrip('('))
                    dataNew[l, 1, m] = dataCol1[l, 0, m]
                    # Right column doesn't need to strip ) since precision limit was 10 and not enough to reach ")"
                    dataNew[l, 2, m] = float(newVal2.rstrip(')'))

            return dataNew

        # Ensure tuple inputs and interpret "*" as all files
        # ensembleFolderPath is a dummy variable here
        self.ensembleFolderPath = self.casePatchFullPaths[0] + self.timesAllRaw[0] + '/'
        self.fileNames = self._ensureTupleInput(fileNames)
        self.ensembleFolderPath = ''
        # Ensure same size as number of files specified
        skipRow = (skipRow,)*len(self.fileNames) if isinstance(skipRow, int) else skipRow
        skipFooter = (skipFooter,)*len(self.fileNames) if isinstance(skipFooter, int) else skipFooter
        # If nTimeSample is -1 or sample interval < 1.5, then use all times
        sampleInterval = 1 if nTimeSample == -1 or nTimeSample > len(self.timesAll)/1.5 else int(np.ceil(len(self.timesAll))/nTimeSample)
        self.sampleTimes = [self.timesAll[0]] if sampleInterval > 1 else self.timesAll
        # Go through all specified properties
        for i in range(len(self.fileNames)):
            # String dtype for left and right column of U so that "(12345" can be read, precision is lstrPrecision and rstrPrecision
            dtype = ('|S' + str(lstrPrecision), float, '|S' + str(rstrPrecision)) if self.fileNames[i] == 'U' else float
            # Go through all patches
            for j in range(len(self.inflowPatches)):
                print('\nReading {}'.format(self.fileNames[i] + ' ' + self.inflowPatches[j]))
                fileNameFullPath = self.casePatchFullPaths[j] + self.timesAllRaw[0] + '/' + self.fileNames[i]
                propertyDictKey = self.fileNames[i] + '_' + self.inflowPatches[j]
                # Initialize first index in the 3rd dimension
                data = np.genfromtxt(fileNameFullPath, skip_header = skipRow[i], skip_footer = skipFooter[i], dtype = dtype)
                # Then go through all times from 2nd time onward
                cnt, milestone = 0, 25
                for k in range(sampleInterval, len(self.timesAll), sampleInterval):
                    # print(self.fileNames[i] + ' ' + self.inflowPatches[j] + ' ' + str(self.timesAll[k]))
                    fileNameFullPath = self.casePatchFullPaths[j] + self.timesAllRaw[k] + '/' + self.fileNames[i]
                    dataPerTime = np.genfromtxt(fileNameFullPath, skip_header = skipRow[i], skip_footer = skipFooter[i], dtype = dtype)
                    data = np.dstack((data, dataPerTime))
                    # Gauge progress
                    cnt += sampleInterval
                    progress = cnt/(len(self.timesAll) + 1)*100.
                    if progress >= milestone:
                        print(' ' + str(milestone) + '%...', end = '')
                        milestone += 25

                # Some postprocessing after reading and dstacking data per time
                data = data.reshape((data.shape[1], data.shape[0], data.shape[2]))
                # If file is U, then strip "(" and ")"
                if self.fileNames[i] == 'U':
                    dataNew = __trimBracketCharacters(data)
                else:
                    dataNew = data

                # Finally, the property data
                self.propertyData[propertyDictKey] = dataNew

        # Collect sample times
        self.sampleTimes = np.empty(dataNew.shape[2])
        i = 0
        for k in range(0, len(self.timesAll), sampleInterval):
            self.sampleTimes[i] = self.timesAll[k]
            i += 1

        # Collect all property keys
        self.propertyKeys = tuple(self.propertyData.keys())
        # Numpy array treatment
        self.sampleTimes = np.array(self.sampleTimes)

        print('\n' + str(self.fileNames) + ' read')


    @timer
    @jit(parallel = True, fastmath = True)
    def calculatePropertyMean(self, startTime = None, stopTime = None, **kwargs):
        # timesAll in _selectTimes() should be sampleTimes in this case, thus temporarily change timesAll to sampleTimes
        timesAllTmp = self.timesAll.copy()
        self.timesAll = self.sampleTimes
        # Find selected times and start, stop indices from sampleTimes
        self.timesSelected, self.startTimeReal, self.stopTimeReal, iStart, iStop = self._selectTimes(startTime = startTime, stopTime = stopTime)
        # Switch timesAll back
        self.timesAll = timesAllTmp
        # Go through all properties
        for i in range(len(self.propertyKeys)):
            # print(i)
            # Selected property data at selected times
            propertySelected = self.propertyData[self.propertyKeys[i]][:, :, iStart:(iStop + 1)]
            # Property mean is sum(property_j*time_j)/sum(times)
            propertyDotTime_sum = 0.
            for j in prange(len(self.timesSelected)):
                # print(j)
                propertyDotTime = np.multiply(propertySelected[:, :, j], self.timesSelected[j])
                propertyDotTime_sum += propertyDotTime

            # Store in dictionary
            self.propertyDataMean[self.propertyKeys[i]] = propertyDotTime_sum/np.sum(self.timesSelected)


    @timer
    @jit(parallel = True)
    def writeMeanToOpenFOAM_Format(self):
        # Go through inflow patches
        for i, patch in enumerate(self.inflowPatches):
            # For each patch, go through (mean) properties
            for j in prange(len(self.propertyKeys)):
                # Pick up only property corresponding current patch
                if patch in self.propertyKeys[j]:
                    # Get property name
                    propertyName = self.propertyKeys[j].replace('_' + patch, '')
                    # Get mean property data
                    propertyDataMean = self.propertyDataMean[self.propertyKeys[j]]
                    # Open file for writing
                    fid = open(self.avgFolderPatchPaths[i] + propertyName, 'w')
                    print('Writing {0} to {1}'.format(propertyName, self.avgFolderPatchPaths[i]))
                    # Define dataType and average (placeholder) value
                    if propertyName in ('k', 'T', 'pd', 'nuSGS', 'kappat'):
                        dataType, average = 'scalar', '0'
                    else:
                        dataType, average = 'vector', '(0 0 0)'

                    # Write the file header
                    fid.write('/*--------------------------------*- C++ -*----------------------------------*\\\n')
                    fid.write('| =========                 |                                                 |\n')
                    fid.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                    fid.write('|  \\\\    /   O peration     | Version:  1.6                                   |\n')
                    fid.write('|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
                    fid.write('|    \\\\/     M anipulation  |                                                 |\n')
                    fid.write('\*---------------------------------------------------------------------------*/\n')
                    fid.write('FoamFile\n')
                    fid.write('{\n')
                    fid.write('    version     2.0;\n')
                    fid.write('    format      ascii;\n')
                    fid.write('    class       ')
                    fid.write(dataType)
                    fid.write('AverageField;\n')
                    fid.write('    object      values;\n')
                    fid.write('}\n')
                    fid.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n')
                    fid.write('// This inflow plane has been averaged from {0} s to {1} s\n'.format(self.startTimeReal, self.stopTimeReal))
                    fid.write('// Average\n')
                    fid.write(average)
                    fid.write('\n\n\n')
                    fid.write(str(propertyDataMean.shape[0]))
                    fid.write('\n')
                    fid.write('(\n')
                    # Write property data
                    for k in range(propertyDataMean.shape[0]):
                        # If U, replace comma with nothing
                        if propertyName == 'U':
                            fid.write(str(tuple(propertyDataMean[k])).replace(',', ''))
                        else:
                            fid.write(str(propertyDataMean[k, 0]))

                        fid.write('\n')
                    fid.write(')')
                    fid.close()













































if __name__ == '__main__':
    from PlottingTool import Plot2D

    caseName = 'ABL_N_L2'
    caseDir = '/media/yluan'
    fileNames = '*'
    nTimeSample = 1000
    startTime, stopTime = 18000, 21000
    case = InflowBoundaryField(caseName = caseName, caseDir = caseDir)
    case.readPropertyData(fileNames = fileNames, nTimeSample = nTimeSample)
    case.calculatePropertyMean(startTime = startTime, stopTime = stopTime)
    case.writeMeanToOpenFOAM_Format()

    # kSouth = case.propertyData['k_south']
    # uSouth = case.propertyData['U_south']






    # caseName = 'ALM_N_H_ParTurb'
    # fileNames = 'Cd'
    # startTime1 = 20000
    # stopTime1 = 22000
    # frameSkip = 182#28
    #
    # turb = TurbineOutputs(caseName = caseName, caseDir = '/media/yluan/Toshiba External Drive')
    #
    # turb.readPropertyData(fileNames = fileNames)
    #
    # turb.calculatePropertyMean(startTime = startTime1, stopTime = stopTime1)
    #
    # listX1 = (turb.timesSelected[::frameSkip],)*3
    # listY1 = (turb.propertyData[fileNames + '_Turb0_Bld0_mean'][::frameSkip],
    #          turb.propertyData[fileNames + '_Turb0_Bld1_mean'][::frameSkip],
    #          turb.propertyData[fileNames + '_Turb0_Bld2_mean'][::frameSkip])
    # listY2 = (turb.propertyData[fileNames + '_Turb1_Bld0_mean'][::frameSkip],
    #           turb.propertyData[fileNames + '_Turb1_Bld1_mean'][::frameSkip],
    #           turb.propertyData[fileNames + '_Turb1_Bld2_mean'][::frameSkip])
    #
    # startTime2 = 21000
    # stopTime2 = 22000
    # turb.calculatePropertyMean(startTime = startTime2, stopTime = stopTime2)
    #
    # listX2 = (turb.timesSelected[::frameSkip],)*3
    # listY3 = (turb.propertyData[fileNames + '_Turb0_Bld0_mean'][::frameSkip],
    #           turb.propertyData[fileNames + '_Turb0_Bld1_mean'][::frameSkip],
    #           turb.propertyData[fileNames + '_Turb0_Bld2_mean'][::frameSkip])
    # listY4 = (turb.propertyData[fileNames + '_Turb1_Bld0_mean'][::frameSkip],
    #           turb.propertyData[fileNames + '_Turb1_Bld1_mean'][::frameSkip],
    #           turb.propertyData[fileNames + '_Turb1_Bld2_mean'][::frameSkip])
    #
    # figDir = '/media/yluan/Toshiba External Drive/' + caseName + '/turbineOutput/Result'
    #
    # # Custom colors
    # colors, _ = Plot2D.setColors()
    #
    # plotsLabel = ('Blade 1', 'Blade 2', 'Blade 3')
    # transparentBg = False
    # xLim1 = (startTime1, stopTime1)
    # yLim = (min(np.min(listY1), np.min(listY2), np.min(listY3), np.min(listY4)), max(np.max(listY1), np.max(listY2), np.max(listY3), np.max(listY4)))
    #
    # show = False
    #
    # clPlot = Plot2D(listY1, listX1, save = True, name = 'Turb0_' + fileNames  + '1', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]', figDir = figDir, xLim = yLim, yLim = xLim1, figWidth = 'half', figHeightMultiplier = 2., show = show, colors = colors[:3][:], gradientBg = True, gradientBgRange = (startTime1, 21800), gradientBgDir = 'y')
    # clPlot.initializeFigure()
    #
    # clPlot.plotFigure(plotsLabel = plotsLabel)
    #
    # clPlot.finalizeFigure(transparentBg = transparentBg)
    #
    # # clPlot2 = Plot2D(listX1, listY2, save = True, name = 'Turb1_' + fileNames  + '1', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]', figDir = figDir, xLim = xLim1, yLim = yLim, figWidth = 'full', show = show, colors = colors[3:6][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # # clPlot2.initializeFigure()
    # # clPlot2.plotFigure(plotsLabel = plotsLabel)
    # # clPlot2.finalizeFigure(transparentBg = transparentBg)
    # #
    # #
    # #
    # #
    # #
    # #
    # # xLim2 = (startTime2, stopTime2)
    # #
    # # show = True
    # #
    # # clPlot = Plot2D(listX2, listY3, save = True, name = 'Turb0_' + fileNames + '2', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]',
    # #                 figDir = figDir, xLim = xLim2, yLim = yLim, figWidth = 'full', show = show, colors = colors[:3][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # # clPlot.initializeFigure()
    # #
    # # clPlot.plotFigure(plotsLabel = plotsLabel)
    # #
    # # clPlot.finalizeFigure(transparentBg = transparentBg)
    # #
    # # clPlot2 = Plot2D(listX2, listY4, save = True, name = 'Turb1_' + fileNames + '2', xLabel = 'Time [s]',
    # #                  yLabel = r'$C_d$ [-]', figDir = figDir, xLim = xLim2, yLim = yLim, figWidth = 'full', show = show, colors = colors[3:6][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # # clPlot2.initializeFigure()
    # # clPlot2.plotFigure(plotsLabel = plotsLabel)
    # # clPlot2.finalizeFigure(transparentBg = transparentBg)
