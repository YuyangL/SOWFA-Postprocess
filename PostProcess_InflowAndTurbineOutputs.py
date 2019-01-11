import os
import numpy as np
from warnings import warn
from Utilities import timer, takeClosest
from numba import njit, jit
from collections.abc import Iterator

class BaseProperties:
    def __init__(self, caseName, caseDir = './', filePre = '', fileSub = '', ensembleFolderName = 'Ensemble', timeCol = 2, resultFolder = 'Result', **kwargs):
        self.caseName, self.caseDir = caseName, caseDir
        self.caseFullPath = caseDir + '/' + caseName + '/'
        # self.startTime, self.stopTime = startTime, stopTime
        self.filePre, self.fileSub = filePre, fileSub
        # Check if result folder is made
        try:
            os.mkdir(self.caseFullPath + resultFolder)
        except OSError:
            pass

        self.ensembleFolderPath = self.caseFullPath + ensembleFolderName + '/'

        self.mergeTimeDirectories(timeCol = timeCol, **kwargs)

        self.timesAll = self.readTime(**kwargs)

        # self.timesSelected, self.startTimeReal, self.stopTimeReal, self.iStart, self.iStop = self.getTimesAndIndices()

        self.propertyData, self.fileNames = {}, []

        print(f'\n{caseName} initialized')


    def ensureTupleInput(self, input):
        inputTuple = (input,) if isinstance(input, (str, np.ndarray)) else input
        # If input[0] is '*' or 'all', get all file names
        inputTuple = os.listdir(self.ensembleFolderPath) if inputTuple[0] in ('*', 'all') else inputTuple
        return inputTuple


    def readTime(self, noRepeat = True):
        # In case of inflow profile
        try:
            timesAll = np.genfromtxt(self.ensembleFolderPath + self.filePre + 'U_mean' + self.fileSub)[:, 0]
        # In case of turbineOutput
        except IOError:
            timesAll = np.genfromtxt(self.ensembleFolderPath + self.filePre + 'Cl' + self.fileSub)[:, 2]

        if noRepeat:
            timesAll = np.unique(timesAll)

        return timesAll


    # @timer
    def mergeTimeDirectories(self, trimOverlapTime = True, timeCol = 2):
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

        # Check whether the directory is made
        try:
            os.mkdir(self.ensembleFolderPath)
        except OSError:
            # If folder not empty, abort
            if os.listdir(self.ensembleFolderPath):
                print(f'\n{self.ensembleFolderPath} files already exist')
                return

        # Available time directories (excluding Ensemble and Result) and file names
        timeDirs = os.listdir(self.caseFullPath)[:-2]
        fileNames = os.listdir(self.caseFullPath + timeDirs[0])
        # Initialize ensemble files
        fileEnsembles = {}
        for fileName in fileNames:
            fileEnsembles[fileName] = open(self.ensembleFolderPath + fileName, "w")

        # Go through time folders and append files to ensemble
        # Excluding Ensemble folder
        for i in range(len(timeDirs)):
            # If trim overlapped time and not in last time directory
            if trimOverlapTime and i < len(timeDirs) - 1:
                try:
                    times = np.genfromtxt(self.caseFullPath + timeDirs[i] + '/' + fileNames[0])[:, timeCol]
                # In case the last line wasn't written properly,
                # which means the simulation was probably aborted, discard the last line
                except ValueError:
                    times = np.genfromtxt(self.caseFullPath + timeDirs[i] + '/' + fileNames[0], skip_footer = 1)[:, timeCol]

                # Index at which trim should start for this file
                iTrim, _ = takeClosest(times, np.float_(timeDirs[i + 1]))

            # Go through each file in this time directory
            for fileName in fileNames:
                # If trim overlapped time and not last time directory and trim is indeed needed
                if trimOverlapTime and i < len(timeDirs) - 1 and iTrim < (len(times) - 1):
                    with open(self.caseFullPath + timeDirs[i] + '/' + fileName, 'r') as file:
                        # Filter out empty lines before iTrim indices can be mapped
                        lines = list(filter(None, (line.rstrip() for line in file)))

                    print(f'\nTrimming overlapped time and adding {fileName} from {timeDirs[i]} to Ensemble...')
                    # Writelines support writing a 1D list, since lines is 2D,
                    # join each row with "\n"
                    fileEnsembles[fileName].writelines("\n".join(lines[:iTrim + 1]))
                # Otherwise, append this file directly to Ensemble
                else:
                    print(f'\nAdding {fileName} from {timeDirs[i]} to Ensemble...')
                    fileEnsembles[fileName].write(open(self.caseFullPath + timeDirs[i] + '/' + fileName).read())

        print("\nMerged time directories for " + str(self.caseName) + " files are stored at:\n " + str(self.ensembleFolderPath))


    # @timer
    @jit
    def getTimesAndIndices(self, startTime = 20000, stopTime = 22000):
        # Bisection left to find actual starting and ending time and their indices
        (iStart, iStop), _ = takeClosest(self.timesAll, (startTime, stopTime))
        # If stopTime larger than any time, iStop = len(timesAll)
        iStop = min(iStop, len(self.timesAll) - 1)
        startTimeReal, stopTimeReal = self.timesAll[iStart], self.timesAll[iStop]

        timesSelected = self.timesAll[iStart:iStop]

        print('\nTime and index information extracted for ' + str(startTimeReal) + ' s and ' + str(stopTimeReal) + ' s')
        return timesSelected, startTimeReal, stopTimeReal, iStart, iStop


    # @timer
    @jit
    def readPropertyData(self, fileNames = ('*',), skipRow = 0, skipCol = 4, verbose = True):
        self.fileNames = self.ensureTupleInput(fileNames)

        for fileName in self.fileNames:
            # Data dictionary of specified property(s) of all times
            self.propertyData[fileName] = \
                np.genfromtxt(self.ensembleFolderPath + self.filePre + fileName + self.fileSub)[skipRow:, skipCol:]

        if verbose:
            print('\n' + str(self.fileNames) + ' read')


    @timer
    @jit
    def calculatePropertyMean(self, axis = 1, startTime = 0, stopTime = 22000):
        self.timesSelected, _, _, iStart, iStop = self.getTimesAndIndices(startTime = startTime, stopTime = stopTime)

        for fileName in self.fileNames:
            self.propertyData[fileName + '_mean'] = np.mean(self.propertyData[fileName][iStart:iStop], axis = axis)



class InflowProfiles(object):
    def __init__(self, caseName, caseDir = './', startTime = 0, stopTime = 1, filePre = '', fileSub = '', fileNameH =
    'hLevelsCell', ensembleFolderName = 'Ensemble_Profiles', **kwargs):
        """
        :param ensembleFolderName: str
        :param caseName: this is a string
        :param caseDir:
        :param startTime:
        :param stopTime:
        :param filePre:
        :param fileSub:
        :param fileNameH:
        """
        self.caseName, self.caseDir = caseName, caseDir
        # Do assert here?
        self.caseFullPath = caseDir + '/' + caseName + '/'
        self.startTime, self.stopTime = startTime, stopTime
        self.filePre, self.fileSub = filePre, fileSub
        self.timeDirs = os.listdir(self.caseFullPath)
        self.ensembleFolderPath = self.caseFullPath + ensembleFolderName + '/'
        self.z = np.genfromtxt(self.caseFullPath + self.timeDirs[0] + '/' + self.filePre + fileNameH + self.fileSub)

        self.startTimeReal, self.stopTimeReal, self.iStart, self.iStop = self.getTimesAndIndices()

        self.propertyData, self.propertyDataStartStop, self.propertyDataMean, self.propertyMeanSpecificZ = {}, {}, {}, {}


    def mergeTimeDirectories(self):
        try:
            os.mkdir(self.ensembleFolderPath)
        except OSError:
            pass

        fileNames = os.listdir(self.caseFullPath + self.timeDirs[0])
        fileEnsembles = {}
        for fileName in fileNames:
            fileEnsembles[fileName] = open(self.ensembleFolderPath + fileName, "w")

        for timeDir in os.listdir(self.caseFullPath)[:-1]:
            for fileName in os.listdir(self.caseFullPath + timeDir):
                fileEnsembles[fileName].write(open(self.caseFullPath + timeDir + '/' + fileName).read())

        print("\nMerged time directories for " + str(self.caseName) + " files are stored at:\n " + str(
                self.ensembleFolderPath))


    def getTimesAndIndices(self):
        dataUall = np.genfromtxt(self.ensembleFolderPath + self.filePre + 'U_mean' + self.fileSub)

        timesAll = dataUall[:, 0]
        startTimeFirstMatch = True
        for i, time in enumerate(timesAll):
            if startTimeFirstMatch and time >= self.startTime:
                iStart, startTimeReal = i, time
                startTimeFirstMatch = False
            elif time >= self.stopTime:
                iStop, stopTimeReal = i, time
                break

        if time < self.stopTime:
            warn('\nSpecified stopTime too large! Using ' + str(time) + ' s instead.', stacklevel = 2)
            iStop, stopTimeReal = i, time

        print('\nTime and index information extracted for ' + str(startTimeReal) + ' s and ' + str(stopTimeReal) + ' s')
        return startTimeReal, stopTimeReal, iStart, iStop


    def getMeanFlowProperty(self, fileNames = ('U_mean', 'V_mean', 'W_mean'), skipCol = 2, specificZ = None):
        """
        :param fileNames:
        :return:
        """
        # if 'iStart' and 'iStop' and 'startTimeReal' and 'stopTimeReal' not in kwargs:
        #     startTimeReal, stopTimeReal, iStart, iStop = getTimesAndIndices(self, ensembleFolderPath)
        # else:
        #     iStart, iStop = kwargs['iStart'], kwargs['iStop']
        #     startTimeReal, stopTimeReal = kwargs['startTimeReal'], kwargs['stopTimeReal']

        # Convert to a tuple if not a tuple to be compatible with iteration below
        if isinstance(fileNames, str):
            fileNames = (fileNames,)

        for fileName in fileNames:
            # Data dictionary of specified property(s) of all times
            self.propertyData[fileName] = \
                np.genfromtxt(self.ensembleFolderPath + self.filePre + fileName + self.fileSub)[:, skipCol:]

            # Data dictionary of specified property(s) of selected times
            self.propertyDataStartStop[fileName] = self.propertyData[fileName][self.iStart:self.iStop]

            propertyDataMeanTmp = np.zeros(self.propertyDataStartStop[fileName].shape[1])
            for iHeight in range(self.propertyDataStartStop[fileName].shape[1]):
                propertyDataMeanTmp[iHeight] = np.mean(self.propertyDataStartStop[fileName][:, iHeight])

            self.propertyDataMean[fileName] = propertyDataMeanTmp

            if isinstance(specificZ, (int, float)):
                from Utilities import takeClosest
                _, specificZloc = takeClosest(self.z, specificZ)
                self.propertyMeanSpecificZ[fileName] = self.propertyDataMean[fileName][specificZloc]

        print('\nMean ' + str(fileNames) + ' of ' + str(self.caseName) + ' computed for the range of ' + str(
                self.startTimeReal)
              + ' to '
              + str(self.stopTimeReal) + ' s')



class TurbineOutputs(BaseProperties):
    def __init__(self, caseName, **kwargs):
        super(TurbineOutputs, self).__init__(caseName + '/turbineOutput', **kwargs)

        self.nTurb, self.nBlade = 0, 0


    @timer
    @jit
    def readPropertyData(self, fileNames = ('*',), skipRow = 0, skipCol = 4, verbose = True):
        super(TurbineOutputs, self).readPropertyData(fileNames = fileNames, skipRow = skipRow, skipCol = skipCol, verbose = False)

        turbInfo = np.genfromtxt(self.ensembleFolderPath + self.filePre + self.fileNames[0] + self.fileSub)[skipRow:, :2]
        # Number of turbines and blades
        self.nTurb, self.nBlade = int(np.max(turbInfo[:, 0]) + 1), int(np.max(turbInfo[:, 1]) + 1)

        fileNamesOld, self.fileNames = self.fileNames, list(self.fileNames)
        for fileName in fileNamesOld:
            # self.fileNamesDetail.append(fileName)
            for i in range(self.nTurb):
                for j in range(self.nBlade):
                    newFileName = fileName + '_Turb' + str(i) + '_Bld' + str(j)
                    self.propertyData[newFileName] = self.propertyData[fileName][(i*self.nBlade + j)::(self.nTurb*self.nBlade), :]
                    self.fileNames.append(newFileName)

        if verbose:
            print('\n' + str(self.fileNames) + ' read')


    # @timer
    # @jit(parallel = True)
    # def calculatePropertyMean(self, axis = 1):
    #     super(TurbineOutputs, self).calculatePropertyMean(axis = axis)























if __name__ is '__main__':
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

    from PlottingTool import Plot2D
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
