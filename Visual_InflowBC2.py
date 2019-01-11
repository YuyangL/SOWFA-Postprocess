# import matplotlib.pyplot as plt
import numpy as np
# from warnings import warn
from PlottingTool_Old import Plot2D
from PostProcess_InflowAndTurbineOutputs import InflowProfiles
from Utilities import readData

zHub = 90


ABL_N_H = InflowProfiles('ABL_N_H', startTime = 14000, stopTime = 18000)

ABL_N_H.getMeanFlowProperty(fileNames = ('U_mean', 'V_mean', 'W_mean'), specificZ = zHub)

Vel = np.sqrt(ABL_N_H.propertyDataMean['U_mean']**2 +
                       ABL_N_H.propertyDataMean['V_mean']**2 +
                       ABL_N_H.propertyDataMean['W_mean']**2)

# VelHub = np.sqrt(ABL_N_H.propertyMeanSpecificZ['U_mean']**2 +
#                        ABL_N_H.propertyMeanSpecificZ['V_mean']**2 +
#                        ABL_N_H.propertyMeanSpecificZ['W_mean']**2)
VelHub = 8.

dataChurchfield = readData('ABL_InflowU_Churchfield.csv', fileDir = './Churchfield', skipRow = 2)
# UUhub_NL, zzi_NL = dataChurchfield['X'], dataChurchfield['Y']
# UUhub_NH, zzi_NH = dataChurchfield['X.1'], dataChurchfield['Y.1']
# UUhub_UL, zzi_UL = dataChurchfield['X.2'], dataChurchfield['Y.2']
# UUhub_UH, zzi_UH = dataChurchfield['X.3'], dataChurchfield['Y.3']

UUhub_NL, zzi_NL = dataChurchfield[:, 0], dataChurchfield[:, 1]
# UUhub_NH, zzi_NH = dataChurchfield['X.1'], dataChurchfield['Y.1']
# UUhub_UL, zzi_UL = dataChurchfield['X.2'], dataChurchfield['Y.2']
# UUhub_UH, zzi_UH = dataChurchfield['X.3'], dataChurchfield['Y.3']


ABL_N_H.getMeanFlowProperty(fileNames = 'T_mean')

ABL_N_H.getMeanFlowProperty(fileNames = 'q3_mean')

ABL_N_H.getMeanFlowProperty(fileNames = ('uu_mean', 'vv_mean', 'ww_mean'))

q3_mean = abs(ABL_N_H.propertyDataMean['q3_mean'])

q3_mean_minLoc = np.argmin(q3_mean)
# zi = ABL_N_H.z[q3_mean_minLoc]

TI = np.divide(np.sqrt(ABL_N_H.propertyDataMean['uu_mean'] +
                       ABL_N_H.propertyDataMean['vv_mean'] +
                       ABL_N_H.propertyDataMean['ww_mean'])/3, Vel)

zLocInversionLow = np.where(ABL_N_H.z >= 700)[0][0] - 1
zLocInversionHi = np.where(ABL_N_H.z >= 800)[0][0]

gapTIold = 0
TIinversion = TI[zLocInversionLow:zLocInversionHi]
for i, val in enumerate(TIinversion[:-1]):
    if abs(TIinversion[i + 1] - TIinversion[i]) > gapTIold:
        gapTIold = abs(TIinversion[i + 1] - TIinversion[i])
        ziLoc = i + zLocInversionLow


zi = (ABL_N_H.z[ziLoc] + ABL_N_H.z[ziLoc + 1])/2

# Plot2D(X = Vel/VelHub, Y = ABL_N_H.z/zi, figDir = './', xLabel = r'$<U>/U_{Hub}$',
#        yLabel = r'$z/z_i$',
# fileName = ABL_N_H.caseName + '_U', xLim = [0, 2], yLim = [0, 1])
#
# Plot2D(X = ABL_N_H.propertyDataMean['T_mean'], Y = ABL_N_H.z, figDir = './', xLabel = r'$<T>$',
#        fileName = ABL_N_H.caseName + '_T')

# Plot2D(plot = 'line', X = ABL_N_H.propertyDataMean['q3_mean'], Y = ABL_N_H.z, figDir = './', \
#                                                                                                xLabel = r'$<q_3>$',
#        fileName = ABL_N_H.caseName + '_q3')

# Plot2D(plot = 'line', X = TI, Y = ABL_N_H.z, figDir = './', \
#                                                                                                xLabel = r'$<TI>$',
#        yLabel = 'z',
#        fileName = ABL_N_H.caseName + '_TI')


# import matplotlib.pyplot as plt

# plt.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
# plt.show()
# Plot2D(X = np.linspace(0, 100, 100), Y = np.linspace(0, 100, 100))

# test.getTimesAndIndices()
#
# test.velocity()

#
# test.get_Zi()
#
#
# test.startTimeReal
#
#
# ensembleFolderPath = test.mergeTimeDirectories()
#
# startTimeReal, stopTimeReal, iStart, iStop = test.getTimesAndIndices(ensembleFolderPath)
#
# dataUmean, dataVmean, dataWmean = \
#     test.velocity(ensembleFolderPath,
#                   iStart = iStart, iStop = iStop, startTimeReal = startTimeReal, stopTimeReal = stopTimeReal)

