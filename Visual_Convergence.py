import numpy as np
from PostProcess_PrecursorAndTurbineOutputs import BaseProperties
from PlottingTool import Plot2D

"""
User Inputs
"""
caseDir = '/media/yluan/Toshiba External Drive/'
# caseName = 'ALM_N_H'
caseName = 'ALM_N_H_ParTurb'
caseName += '/Probes'
fileNames = 'uuPrime2'  # ('uuPrime2', 'UAvg', 'Rmean')
nProbe = 4
# Which column is time
timeCol = 0
# Characters to remove before reading data into numpy arrays
invalidChars = ('(', ')')
times = (20500, 22000)

"""
Plot Settings
"""
saveFig = True
xLabel = 'Time [s]'
# yLabel = r'$R_{11}$'
lineLabels = ('Turb0 + $1D$', 'Turb0 + $3D$',
              'Turb1 + $1D$', 'Turb1 + $3D$')
xLim = times
figWidth = 'half'
show = False
gradientBg, gradientBgRange = True, (20000, 21600)


"""
Merge Time Directories and Read Properties
"""
probe = BaseProperties(caseName = caseName, caseDir = caseDir, timeCols = timeCol, forceRemerge = False)

# From now on all files are read through Ensemble regardless
# Trim invalid '(' and ')' characters
probe.trimInvalidCharacters(fileNames = fileNames, invalidChars = invalidChars)

# Read properties from Ensemble
probe.readPropertyData(fileNames = fileNames, skipCol = timeCol + 1)

# # Calculate mean
# probe.calculatePropertyMean()


"""
Regroup Data Into xx, yy, zz or x, y, z Categories and Plot
"""
# Function to decompose data into xx, yy, zz or x, y, z components and plot
def decomposeDataAndPlot(step, subscript, startCol, fileName):
    # For xx component
    propertyDecomp = {}
    listY = []
    for i in range(nProbe):
        propertyDecomp['probe' + str(i)] = probe.propertyData[fileName][:, startCol + step*i]
        listY.append(propertyDecomp['probe' + str(i)])

    yLim = (np.min(listY), np.max(listY))

    plot = Plot2D(listX, listY, save = saveFig, name = fileName + '_' + subscript + '_Convergence', xLabel = xLabel,
                  yLabel = yLabel,
                  figDir = figDir, xLim = xLim, yLim = yLim, figWidth = figWidth, show = show, gradientBg = gradientBg)

    plot.initializeFigure()
    plot.plotFigure(plotsLabel = lineLabels)
    plot.finalizeFigure(transparentBg = False)


figDir = probe.caseFullPath + 'Result'
listX = [probe.timesSelected]*nProbe
if 'uuPrime2' in fileNames:
    fileName = 'uuPrime2'
    # Pick up columns every 6 steps due to symmetric tensor saved in the order of
    # (xx, xy, xz, yy, yz, zz)
    step, startCol = 6, 0

    # Decompose and plot xx component
    yLabel = r'$R_{11}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '11', startCol = startCol, fileName = fileName)

    # Decompose and plot xy component
    yLabel = r'$R_{12}$ [m$^2$/s$^2$]'
    startCol = 1
    decomposeDataAndPlot(step = step, subscript = '12', startCol = startCol, fileName = fileName)

    # Decompose and plot xz component
    yLabel = r'$R_{13}$ [m$^2$/s$^2$]'
    startCol = 2
    decomposeDataAndPlot(step = step, subscript = '13', startCol = startCol, fileName = fileName)

    # Decompose and plot yy component
    startCol = 3
    yLabel = r'$R_{22}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '22', startCol = startCol, fileName = fileName)

    # Decompose and plot yz component
    yLabel = r'$R_{23}$ [m$^2$/s$^2$]'
    startCol = 4
    decomposeDataAndPlot(step = step, subscript = '23', startCol = startCol, fileName = fileName)

    # Decompose and plot yy component
    startCol = 5
    yLabel = r'$R_{33}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '33', startCol = startCol, fileName = fileName)



if 'UAvg' in fileNames:
    fileName = 'UAvg'
    step, startCol = 3, 0

    # Decompose and plot x component
    yLabel = r'$\langle U_{x} \rangle$ [m/s]'
    decomposeDataAndPlot(step = step, subscript = '1', startCol = startCol, fileName = fileName)

    # Decompose and plot y component
    startCol = 1
    yLabel = r'$\langle U_{y} \rangle$ [m/s]'
    decomposeDataAndPlot(step = step, subscript = '2', startCol = startCol, fileName = fileName)

    # Decompose and plot z component
    startCol = 2
    yLabel = r'$\langle U_{z} \rangle$ [m/s]'
    decomposeDataAndPlot(step = step, subscript = '3', startCol = startCol, fileName = fileName)


if 'Rmean' in fileNames:
    fileName = 'Rmean'
    # Pick up columns every 6 steps due to symmetric tensor saved in the order of
    # (xx, xy, xz, yy, yz, zz)
    step, startCol = 6, 0

    # Decompose and plot xx component
    yLabel = r'$R_{11}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '11', startCol = startCol, fileName = fileName)

    # Decompose and plot yy component
    startCol = 3
    yLabel = r'$R_{22}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '22', startCol = startCol, fileName = fileName)

    # Decompose and plot yy component
    startCol = 5
    yLabel = r'$R_{33}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '33', startCol = startCol, fileName = fileName)










