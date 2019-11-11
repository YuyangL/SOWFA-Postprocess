from PrecursorAndTurbineOutputs import TurbineOutputs
from PlottingTool import Plot2D
import numpy as np
import matplotlib.pyplot as plt

"""
Calculate Power Ratio Between Downwind and Upwind Turbines
"""
caseDir = '/media/yluan/Toshiba External Drive/'
caseName = 'ALM_N_H'
propertyNames = 'powerGenerator'
times = (20500, 22000)
# Initialize turb object
turb = TurbineOutputs(caseName = caseName, caseDir = caseDir, forceRemerge = False)

# Read property data
turb.readPropertyData(fileNames = propertyNames)

# Calculate time averaged mean between times, thus axis is row
turb.calculatePropertyMean(startTime = times[0], stopTime = times[1], axis = 'row')

powerRatio = turb.propertyData[propertyNames + '_Turb1_mean']/turb.propertyData[propertyNames + '_Turb0_mean']

print(f'\nPower ratio between downwind front turbine and upwind turbine is {powerRatio}')



# """
# User Inputs
# """
# caseDir = '/media/yluan/Toshiba External Drive/'
# caseNames = ('ALM_N_H', 'ALM_N_H_ParTurb')
# propertyNames = ('Cl', 'Cd')
# xLabels = {propertyNames[0]: r'$C_l$ [-]',
#            propertyNames[1]: r'$C_d$ [-]'}
# times = (20000, 22000)  # s
# # Turbines are 9.1552 rpm => 6.554 s/r. Time step is 0.036 s/step => 182.046 steps/r
# frameSkip = 182  # steps/r
# # First 4 columns are not property data; next 6 columns are 0 for Cl and Cd
# propertyColSkip = 10  # Default 4
#
# """
# Plot Settings
# """
# # Figure width is half of A4 page, height is multiplied to elongate it
# figWidth, figHeightMultiplier = 'half', 2.
# # Custom colors of different turbines
# colors, _ = Plot2D.setColors()
#
# plotsLabel = ('Blade 1', 'Blade 2', 'Blade 3')
# show, saveFig, transparentBg = False, True, False
# yLim = (times[0], times[1])
# yLabel = 'Time [s]'
# gradientBg, gradientBgDir, gradientBgRange = True, 'y', (times[0], times[0] + 0.8*(times[1] - times[0]))
#
# """
# Read Property Data and Plot
# """
# # Go through cases
# for caseName in caseNames:
#     figDir = caseDir + caseName + '/turbineOutput/Result'
#     # Initialize turb object
#     turb = TurbineOutputs(caseName = caseName, caseDir = caseDir)
#
#     # Read property data
#     turb.readPropertyData(fileNames = propertyNames, skipCol = propertyColSkip)
#
#     # Calculate ALM segment averaged mean between times
#     turb.calculatePropertyMean(startTime = times[0], stopTime = times[1])
#
#     # Actual times in a list of length of number of lines in a figure
#     listY = (turb.timesSelected[::frameSkip],)*turb.nBlade
#     # Go through properties
#     for propertyName in propertyNames:
#         # Properties dictionary in a list of length of number of lines in a figure
#         listX = {}
#         # Initialize extreme xLim that gets updated in the following turbine loop
#         xLim = (1e9, -1e9)
#         # Go through turbines
#         for i in range(turb.nTurb):
#             meanPropertyName = propertyName + '_Turb' + str(i)
#             listX[str(i)] = (turb.propertyData[meanPropertyName + '_Bld0_mean'][::frameSkip],
#                              turb.propertyData[meanPropertyName + '_Bld1_mean'][::frameSkip],
#                              turb.propertyData[meanPropertyName + '_Bld2_mean'][::frameSkip])
#             # xLim gets updated every turbine so all turbines are considered in the end
#             xLim = (min(xLim[0], np.min(listX[str(i)])), max(xLim[1], np.max(listX[str(i)])))
#
#         """
#         Plot for every turbine for this property
#         """
#         for i in range(turb.nTurb):
#             # Initialize figure object
#             # In this custom color, blades of different turbines are of different colors
#             propertyPlot = Plot2D(listX[str(i)], listY, save = saveFig, name = 'Turb' + str(i) + '_' + propertyName, xLabel = xLabels[propertyName], yLabel = yLabel, figDir = figDir, xLim = xLim, yLim = yLim, figWidth = figWidth, figHeightMultiplier = figHeightMultiplier, show = show, colors = colors[i*turb.nBlade:(i + 1)*turb.nBlade][:], gradientBg = gradientBg, gradientBgRange = gradientBgRange, gradientBgDir = gradientBgDir)
#
#             # Create the figure window
#             propertyPlot.initializeFigure()
#
#             # Plot the figure
#             propertyPlot.plotFigure(plotsLabel = plotsLabel)
#
#             # Finalize figure
#             propertyPlot.finalizeFigure(transparentBg = transparentBg)
#
#             # Close current figure window
#             # so that the next figure will be based on a new figure window even if the same name
#             plt.close()








