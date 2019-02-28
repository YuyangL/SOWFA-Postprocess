import matplotlib.pyplot as plt
import numpy as np
# from warnings import warn
from PostProcess_PrecursorAndTurbineOutputs import InflowProfiles, BoundaryLayerProperties
from PlottingTool import Plot2D
from Utilities import readData
from numba import prange

"""
User Inputs
"""
# caseDir, caseName = 'J:', 'Doekemeijer/neutral_3kmx3kmx1km_lowTI_225deg_11mps'
caseDir, caseName = 'J:', 'ABL_N_H_HiSpeed'
# Profile of interest
profile = 'TI'  # 'U', 'T', 'heatFlux', 'TI'
startTime, stopTime = 18000, 22000
# Hub height velocity, hub height, rotor D
Uhub, zHub, D = 10., 90., 126.
# Inversion layer height, inversion layer width
zi, invW = 750., 100.
# Whether to calculate zi from q3_mean and Tw_mean, will override existing zi
calc_zi = False
# Force to remerge all time directories to Ensemble folder?
forceRemerge = False  # [CAUTION]
# Unused
startTime2, stopTime2 = 3000, 6000


"""
Plot Settings
"""
show, save = True, True
# Fill in area alpha for rotor swept area and capping inversion layer
fillAlpha = 0.25


"""
Process User Inputs
"""
if profile == 'U':
    fileNames = ['U_mean', 'V_mean', 'W_mean']
elif profile == 'T':
    fileNames = ['T_mean']
elif profile == 'heatFlux':
    fileNames = ['q3_mean', 'Tw_mean']
elif profile == 'TI':
    fileNames = ['uu_mean', 'vv_mean', 'ww_mean']

# q3_mean and Tw_mean are for zi calculation, don't add again for the 'heatFlux' profile
fileNames += ['q3_mean', 'Tw_mean'] if calc_zi and profile != 'heatFlux' else []
# Reference data directory for U from Churchfield et al. 2012
refDataDir = caseDir + 'Churchfield/'
if caseName in ('ABL_N_L', 'ABL_N_L2'):
    refDataName, caseName2 = 'InflowU_N_L', 'ABL-N-L'
elif caseName == 'ABL_N_H':
    refDataName, caseName2 = 'InflowU_N_H', 'ABL-N-H'
elif caseName == 'ABL_Uns_L':
    refDataName, caseName2 = 'InflowU_Uns_L', 'ABL-Uns-L'
elif caseName == 'ABL_Uns_H':
    refDataName, caseName2 = 'InflowU_Uns_H', 'ABL-Uns-H'
else:
    refDataName, caseName2 = '', r'N-Low $I$-225^{\circ}-11 m/s'

refDataName += '.csv'


"""
Read and Process BL Data
"""
# Initialize bl object
bl = BoundaryLayerProperties(caseName = caseName, caseDir = caseDir, forceRemerge = forceRemerge)
# Read data for given fileNames
bl.readPropertyData(fileNames = fileNames)
# Calculate the temporal average for given startTime and stopTime
bl.calculatePropertyMean(startTime = startTime, stopTime = stopTime)

if calc_zi or profile == 'heatFlux':
    # Calculate zi by finding the minimum z-direction T flux
    # According to statisticsABL.H, q3_mean correspdonds to sgsTempFluxLevelsList.z(), Tw_mean corresponds to tempFluxLevelsList.z()
    minTflux, TfluxLst = 1e20, np.empty_like(bl.hLvls)
    # Go through all height levels
    for i in prange(len(bl.hLvls)):
        # Tflux = bl.propertyData['Tw_mean_mean'][i]
        if i == 0 or i == len(bl.hLvls) - 1:
            Tflux = bl.propertyData['q3_mean_mean'][i]
        else:
            Tflux = bl.propertyData['q3_mean_mean'][i] + 0.5*(bl.propertyData['Tw_mean_mean'][i] + bl.propertyData['Tw_mean_mean'][i - 1])

        # Append Tflux of each height to list for plotting the 'heatFlux' profile
        TfluxLst[i] = Tflux
        if Tflux < minTflux and calc_zi:
            zi, minTflux = bl.hLvls[i], Tflux

figDir = bl.resultDir

# For horizontal U profile
if profile == 'U':
    """
    Plot U BL Profile
    """
    uMean = bl.propertyData['U_mean_mean']
    vMean = bl.propertyData['V_mean_mean']
    wMean = bl.propertyData['W_mean_mean']
    # uvMean = np.sqrt(uMean**2 + vMean**2)
    uvwMean = np.sqrt(uMean**2 + vMean**2 + wMean**2)
    # Reference data, 1st column is x, 2nd column is y
    refData = readData(refDataName, fileDir = refDataDir, skipRow = 0)
    # Since refData is sorted from low x to high x, what we want is from low y to high y
    # Thus sort the 2nd column
    refData = refData[refData[:, 1].argsort()]
    # Add both normalized simulation and reference data to lists to plot
    xList, yList = [uvwMean/Uhub, refData[:, 0]], [bl.hLvls/zi, refData[:, 1]]
    # X, Y limit to be inline with Churchfield's data
    xLim, yLim = (0, 2), (0, 1)
    # Initialize figure object
    plot = Plot2D(xList, yList, xLabel = r'$\langle \overline{U}\rangle /U_{\rm{hub}}$', yLabel = r'$z/z_i$', figDir = figDir, name = profile, save = save, show = show, xLim = xLim, yLim = yLim)
    plot.initializeFigure()
    # Fill in the rotor swept area
    plot.axes[0].fill_between(xLim, ((zHub + D/2)/zi,)*2, ((zHub - D/2)/zi,)*2, alpha = 0.25, facecolor =
    plot.colors[2], zorder = -1)
    plot.axes[0].fill_between(xLim, ((zi + 0.5*invW)/zi,)*2, ((zi - 0.5*invW)/zi,)*2, alpha = fillAlpha,
                              facecolor = plot.colors[3], zorder = -2)
    # Plot figure
    plot.plotFigure(plotsLabel = (caseName2, 'Churchfield et al.'))

elif profile == 'T':
    """
    Plot T BL Profile
    """
    x = bl.propertyData['T_mean_mean']
    xLim = (x.min()*0.99, x.max()*1.01)
    plot = Plot2D(x, bl.hLvls, xLabel = r'$\langle \overline{\Theta}\rangle$ [K]', yLabel = r'$z$ [m]',
                  figDir = figDir, name = profile, save = save, show = show, xLim = xLim)
    plot.initializeFigure()
    plot.plotFigure()

elif profile == 'heatFlux':
    """
    Plot Heat Flux BL Profile
    """
    x = TfluxLst
    # x.min() < 0, thus *1.1
    xLim = (x.min()*1.1, x.max()*1.1)
    plot = Plot2D(x, bl.hLvls, xLabel = r'$\langle \overline{q_z}\rangle$ [W/m$^2$]',
                  yLabel = r'$z$ [m]',
                  figDir = figDir, name = profile, save = save, show = show, xLim = xLim)
    plot.initializeFigure()
    plot.plotFigure()
    # Limit the number of x ticks since x tick labels are really long
    plt.locator_params(axis = 'x', nbins = 6)

elif profile == 'TI':
    """
    Plot TI BL Profile
    """
    # Reference TI [%] from Churchfield et al. 2012 at rotor bottom, hub, and apex
    if caseName2 == 'ABL-N-L':
        refData = np.array([[5.9, zHub - 0.5*D],
                            [4.9, zHub],
                            [4.4, zHub + 0.5*D]])
    elif caseName2 == 'ABL-N-H':
        refData = np.array([[10.0, zHub - 0.5*D],
                            [8.4, zHub],
                            [7.6, zHub + 0.5*D]])
    # In case not reference data availalble
    else:
        refData = None

    # TI in percentage and normalized by hub height U
    ti = np.sqrt((bl.propertyData['uu_mean_mean'] + bl.propertyData['vv_mean_mean'] + bl.propertyData['ww_mean_mean'])/3.)/Uhub*100.
    xList, yList = ((ti, refData[:, 0]), (bl.hLvls, refData[:, 1])) if refData is not None else (ti, bl.hLvls)
    xLim = (ti.min()*0.9, ti.max()*1.1)
    plot = Plot2D(xList, yList, xLabel = r'$\langle \overline{I}\rangle$ [\%]',
                  yLabel = r'$z$ [m]',
                  figDir = figDir, name = profile, save = save, show = show, xLim = xLim, type = ('line', 'scatter'))
    plot.initializeFigure()
    plot.plotFigure(plotsLabel = (caseName2, 'Churchfield et al.'))
    # # Plot reference data from Churchfield et al. 2012
    # plot.axes[0].scatter(refData[:, 0], refData[:, 1], colo)


# Fill in rotor swept area as well as capping inversion layer for non-normalized plots
if profile != 'U':
    plot.axes[0].fill_between(xLim, ((zHub + D/2),)*2, ((zHub - D/2),)*2, alpha = fillAlpha, facecolor =
    plot.colors[2], zorder = -1)
    plot.axes[0].fill_between(xLim, (zi + 0.5*invW,)*2, (zi - 0.5*invW,)*2, alpha = fillAlpha, facecolor = plot.colors[3], zorder = -2)

plot.finalizeFigure()








# zHub = 90
#
#
# ABL_N_H = InflowProfiles('ABL_N_H', startTime = 14000, stopTime = 18000)
#
# ABL_N_H.getMeanFlowProperty(fileNames = ('U_mean', 'V_mean', 'W_mean'), specificZ = zHub)
#
# Vel = np.sqrt(ABL_N_H.propertyDataMean['U_mean']**2 +
#                        ABL_N_H.propertyDataMean['V_mean']**2 +
#                        ABL_N_H.propertyDataMean['W_mean']**2)
#
# # VelHub = np.sqrt(ABL_N_H.propertyMeanSpecificZ['U_mean']**2 +
# #                        ABL_N_H.propertyMeanSpecificZ['V_mean']**2 +
# #                        ABL_N_H.propertyMeanSpecificZ['W_mean']**2)
# VelHub = 8.
#
# dataChurchfield = readData('ABL_InflowU_Churchfield.csv', fileDir = './Churchfield', skipRow = 2)
# # UUhub_NL, zzi_NL = dataChurchfield['X'], dataChurchfield['Y']
# # UUhub_NH, zzi_NH = dataChurchfield['X.1'], dataChurchfield['Y.1']
# # UUhub_UL, zzi_UL = dataChurchfield['X.2'], dataChurchfield['Y.2']
# # UUhub_UH, zzi_UH = dataChurchfield['X.3'], dataChurchfield['Y.3']
#
# UUhub_NL, zzi_NL = dataChurchfield[:, 0], dataChurchfield[:, 1]
# # UUhub_NH, zzi_NH = dataChurchfield['X.1'], dataChurchfield['Y.1']
# # UUhub_UL, zzi_UL = dataChurchfield['X.2'], dataChurchfield['Y.2']
# # UUhub_UH, zzi_UH = dataChurchfield['X.3'], dataChurchfield['Y.3']
#
#
# ABL_N_H.getMeanFlowProperty(fileNames = 'T_mean')
#
# ABL_N_H.getMeanFlowProperty(fileNames = 'q3_mean')
#
# ABL_N_H.getMeanFlowProperty(fileNames = ('uu_mean', 'vv_mean', 'ww_mean'))
#
# q3_mean = abs(ABL_N_H.propertyDataMean['q3_mean'])
#
# q3_mean_minLoc = np.argmin(q3_mean)
# # zi = ABL_N_H.z[q3_mean_minLoc]
#
# TI = np.divide(np.sqrt(ABL_N_H.propertyDataMean['uu_mean'] +
#                        ABL_N_H.propertyDataMean['vv_mean'] +
#                        ABL_N_H.propertyDataMean['ww_mean'])/3, Vel)
#
# zLocInversionLow = np.where(ABL_N_H.z >= 700)[0][0] - 1
# zLocInversionHi = np.where(ABL_N_H.z >= 800)[0][0]
#
# gapTIold = 0
# TIinversion = TI[zLocInversionLow:zLocInversionHi]
# for i, val in enumerate(TIinversion[:-1]):
#     if abs(TIinversion[i + 1] - TIinversion[i]) > gapTIold:
#         gapTIold = abs(TIinversion[i + 1] - TIinversion[i])
#         ziLoc = i + zLocInversionLow
#
#
# zi = (ABL_N_H.z[ziLoc] + ABL_N_H.z[ziLoc + 1])/2






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

