import os
import numpy as np
from warnings import warn

class InflowProfiles(object):
    def __init__(self, caseName, caseDir = './', startTime = 0, stopTime = 1, filePre = '', fileSub = '', fileNameH =
    'hLevelsCell', ensembleFolderName = 'Ensemble_Profiles'):
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
