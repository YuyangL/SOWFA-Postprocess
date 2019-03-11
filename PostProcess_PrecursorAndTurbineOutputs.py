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
        inputTuple = os.listdir(self.ensembleFolderPath) if inputTuple[0] in ('*', 'all') else inputTuple
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
        stopTime = self.timesAll[1] if stopTime is None else stopTime
        # Bisection left to find actual starting and ending time and their indices
        (iStart, iStop) = np.searchsorted(self.timesAll, (startTime, stopTime))
        # If stopTime larger than any time, iStop = len(timesAll)
        iStop = min(iStop, len(self.timesAll) - 1)
        startTimeReal, stopTimeReal = self.timesAll[iStart], self.timesAll[iStop]

        timesSelected = self.timesAll[iStart:(iStop + 1)]

        # print('\nTime and index information extracted for ' + str(startTimeReal) + ' s - ' + str(stopTimeReal) + ' s')
        return timesSelected, startTimeReal, stopTimeReal, iStart, iStop


    @jit(parallel = True)
    def readPropertyData(self, fileNames = ('*',), skipRow = 0, skipCol = 0):
        self.fileNames = self._ensureTupleInput(fileNames)
        skipRow = (skipRow,)*len(self.fileNames) if isinstance(skipRow, int) else skipRow
        skipCol = (skipCol,)*len(self.fileNames) if isinstance(skipCol, int) else skipCol

        for i in prange(len(self.fileNames)):
            # Data dictionary of specified property(s) of all times
            self.propertyData[self.fileNames[i]] = \
                np.genfromtxt(self.ensembleFolderPath + self.filePre + self.fileNames[i] + self.fileSub)[skipRow[i]:,
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


class BoundaryLayerProperties(BaseProperties):
    def __init__(self, caseName, fileNameH = 'hLevelsCell', blFolder = 'ABL', **kwargs):
        self.fileNameH = fileNameH
        super(BoundaryLayerProperties, self).__init__(caseName = caseName + '/' + blFolder, timeCols = 0, excludeFile =
        fileNameH, **kwargs)
        # Copy fileNameH to Ensemble in order to use it later
        time = os.listdir(self.caseFullPath)[0]
        shutil.copy2(self.caseFullPath + time + '/' + fileNameH, self.ensembleFolderPath)


    def readPropertyData(self, fileNames = ('*'), **kwargs):
        # Read height levels
        self.hLvls = np.genfromtxt(self.ensembleFolderPath + self.fileNameH)
        # Override skipCol to suit inflow property files
        # Columns to skip are 0: time; 1: time step
        super(BoundaryLayerProperties, self).readPropertyData(fileNames = fileNames, skipCol = 2)


    def calculatePropertyMean(self, startTime = None, stopTime = None, **kwargs):
        # Override axis to suit inflow property files
        super(BoundaryLayerProperties, self).calculatePropertyMean(axis = 0, startTime = startTime, stopTime = stopTime)


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


    # @timer
    # @jit(parallel = True)
    # def calculatePropertyMean(self, axis = 1):
    #     super(TurbineOutputs, self).calculatePropertyMean(axis = axis)























if __name__ is '__main__':
    from PlottingTool import Plot2D

    caseName = 'ALM_N_H_ParTurb'
    fileNames = 'Cd'
    startTime1 = 20000
    stopTime1 = 22000
    frameSkip = 182#28

    turb = TurbineOutputs(caseName = caseName, caseDir = '/media/yluan/Toshiba External Drive')

    turb.readPropertyData(fileNames = fileNames)

    turb.calculatePropertyMean(startTime = startTime1, stopTime = stopTime1)

    listX1 = (turb.timesSelected[::frameSkip],)*3
    listY1 = (turb.propertyData[fileNames + '_Turb0_Bld0_mean'][::frameSkip],
             turb.propertyData[fileNames + '_Turb0_Bld1_mean'][::frameSkip],
             turb.propertyData[fileNames + '_Turb0_Bld2_mean'][::frameSkip])
    listY2 = (turb.propertyData[fileNames + '_Turb1_Bld0_mean'][::frameSkip],
              turb.propertyData[fileNames + '_Turb1_Bld1_mean'][::frameSkip],
              turb.propertyData[fileNames + '_Turb1_Bld2_mean'][::frameSkip])

    startTime2 = 21000
    stopTime2 = 22000
    turb.calculatePropertyMean(startTime = startTime2, stopTime = stopTime2)

    listX2 = (turb.timesSelected[::frameSkip],)*3
    listY3 = (turb.propertyData[fileNames + '_Turb0_Bld0_mean'][::frameSkip],
              turb.propertyData[fileNames + '_Turb0_Bld1_mean'][::frameSkip],
              turb.propertyData[fileNames + '_Turb0_Bld2_mean'][::frameSkip])
    listY4 = (turb.propertyData[fileNames + '_Turb1_Bld0_mean'][::frameSkip],
              turb.propertyData[fileNames + '_Turb1_Bld1_mean'][::frameSkip],
              turb.propertyData[fileNames + '_Turb1_Bld2_mean'][::frameSkip])

    figDir = '/media/yluan/Toshiba External Drive/' + caseName + '/turbineOutput/Result'

    # Custom colors
    colors, _ = Plot2D.setColors()

    plotsLabel = ('Blade 1', 'Blade 2', 'Blade 3')
    transparentBg = False
    xLim1 = (startTime1, stopTime1)
    yLim = (min(np.min(listY1), np.min(listY2), np.min(listY3), np.min(listY4)), max(np.max(listY1), np.max(listY2), np.max(listY3), np.max(listY4)))

    show = False

    clPlot = Plot2D(listY1, listX1, save = True, name = 'Turb0_' + fileNames  + '1', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]', figDir = figDir, xLim = yLim, yLim = xLim1, figWidth = 'half', figHeightMultiplier = 2., show = show, colors = colors[:3][:], gradientBg = True, gradientBgRange = (startTime1, 21800), gradientBgDir = 'y')
    clPlot.initializeFigure()

    clPlot.plotFigure(plotsLabel = plotsLabel)

    clPlot.finalizeFigure(transparentBg = transparentBg)

    # clPlot2 = Plot2D(listX1, listY2, save = True, name = 'Turb1_' + fileNames  + '1', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]', figDir = figDir, xLim = xLim1, yLim = yLim, figWidth = 'full', show = show, colors = colors[3:6][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # clPlot2.initializeFigure()
    # clPlot2.plotFigure(plotsLabel = plotsLabel)
    # clPlot2.finalizeFigure(transparentBg = transparentBg)
    #
    #
    #
    #
    #
    #
    # xLim2 = (startTime2, stopTime2)
    #
    # show = True
    #
    # clPlot = Plot2D(listX2, listY3, save = True, name = 'Turb0_' + fileNames + '2', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]',
    #                 figDir = figDir, xLim = xLim2, yLim = yLim, figWidth = 'full', show = show, colors = colors[:3][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # clPlot.initializeFigure()
    #
    # clPlot.plotFigure(plotsLabel = plotsLabel)
    #
    # clPlot.finalizeFigure(transparentBg = transparentBg)
    #
    # clPlot2 = Plot2D(listX2, listY4, save = True, name = 'Turb1_' + fileNames + '2', xLabel = 'Time [s]',
    #                  yLabel = r'$C_d$ [-]', figDir = figDir, xLim = xLim2, yLim = yLim, figWidth = 'full', show = show, colors = colors[3:6][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # clPlot2.initializeFigure()
    # clPlot2.plotFigure(plotsLabel = plotsLabel)
    # clPlot2.finalizeFigure(transparentBg = transparentBg)
