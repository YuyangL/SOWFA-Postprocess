import numpy as np
from PostProcess_PrecursorAndTurbineOutputs import BaseProperties
from PlottingTool import Plot2D
from numba import prange

"""
User Inputs
"""
caseDir = 'J:'
# caseName = 'ALM_N_H'
caseName = 'ALM_N_H_OneTurb'
# caseName += '/Probes'
fileNames = 'UAvg'  # ('uuPrime2', 'UAvg', 'Rmean')
# Which column is time
timeCol = 0  # 'infer'
# Characters to remove before reading data into numpy arrays
invalidChars = ('(', ')')
times = (20500, 23000)

"""
Plot Settings
"""
saveFig = True
xLabel = 'Time [s]'
# yLabel = r'$R_{11}$'
xLim = times
figWidth = 'half'
show = False
gradientBg, gradientBgRange = True, (times[0], times[1])


"""
Process User Inputs
"""
if caseName == 'ALM_N_H_OneTurb':
    nProbe = 8
else:
    nProbe = 4

if caseName == 'ALM_N_H_ParTurb':
    lineLabels = ('Turb0 + $1D$', 'Turb0 + $3D$',
                  'Turb1 + $1D$', 'Turb1 + $3D$')
elif caseName == 'ALM_N_H_OneTurb':
    lineLabels = (r'Hub $-3D$', r'Hub $-1D$', r'Hub $+1D$', r'Hub $+2D$', r'Hub $+4D$', r'Apex $-1D$', r'Apex $+2D$', r'Apex $+4D$')


"""
Merge Time Directories and Read Properties
"""
probe = BaseProperties(caseName = caseName + '/Probes', caseDir = caseDir, timeCols = timeCol, forceRemerge = False)

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

    # listY = np.array(listY)
    if caseName == 'ALM_N_H_OneTurb':
        listY2 = [listY[3], listY[5], listY[7]]
        del listY[7]
        del listY[5]
        del listY[3]
        listY += listY2

    nPlot = 2 if caseName == 'ALM_N_H_OneTurb' else 1
    nProbePlot = (0, 5, nProbe) if caseName == 'ALM_N_H_OneTurb' else (0, nProbe)
    for i in prange(nPlot):
        yLim = (np.min(listY) - np.abs(np.min(listY)*0.05),
                np.max(listY) + np.abs(np.max(listY)*0.05))
        plot = Plot2D(listX[nProbePlot[i]:nProbePlot[i + 1]], listY[nProbePlot[i]:nProbePlot[i + 1]], save = saveFig, name = fileName + '_' + subscript + '_Convergence_' + str(i), xLabel = xLabel,
                      yLabel = yLabel,
                      figDir = figDir, xLim = xLim, yLim = yLim, figWidth = figWidth, show = show, gradientBg = gradientBg)

        plot.initializeFigure()
        plot.plotFigure(plotsLabel = lineLabels[nProbePlot[i]:nProbePlot[i + 1]])
        plot.finalizeFigure()


figDir = probe.caseFullPath + 'Result'
# listX = [probe.timesSelected]*nProbe
listX = (probe.timesAll,)*nProbe
if 'uuPrime2' in fileNames:
    """
    Reynolds Stress Plots
    """
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
    """
    UAvg Plots
    """
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
    """
    SFS Deviatoric Stress Plots
    """
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










