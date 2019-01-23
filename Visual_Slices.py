"""
Read and Visualize Horizontal/Vertical Slices in 2/3D
"""
import numpy as np
import os
from numba import njit, jit
from Utilities import timer
from scipy.interpolate import griddata
from PlottingTool import Plot2D_InsetZoom, PlotSurfaceSlices3D

caseDir = '/media/yluan/Toshiba External Drive/'
caseName = 'ALM_N_H'
time = 22000.0918025
# sliceNames = ['alongWind', 'groundHeight', 'hubHeight', 'oneDaboveHubHeight', 'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
sliceNames = ['oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# sliceNames = ['alongWind']
propertyName = 'uuPrime2'
fileExt = '.raw'
# Combine propertyName with sliceNames and Subscript to form the full file names
# Don't know why I had to copy it...
fileNames = sliceNames.copy()
for i, name in enumerate(sliceNames):
    sliceNames[i] = propertyName + '_' + name + '_Slice'
    fileNames[i] = sliceNames[i] + fileExt

precisionX, precisionY, precisionZ = 1500j, 1500j, 500j
interpMethod = 'linear'

"""
Plot Settings
"""
figDir = caseDir + caseName + '/Slices/Result/' + str(time)
try:
    os.mkdir(figDir)
except FileExistsError:
    pass

xLim, yLim, zLim = (0, 3000), (0, 3000), (0, 1000)
show, save = False, True
xLabel, yLabel, zLabel = r'$x$ [m]', r'$y$ [m]', r'$z$ [m]'
valLabels = (r'$b_{11}$ [-]', r'$b_{12}$ [-]', r'$b_{13}$ [-]', r'$b_{22}$ [-]', r'$b_{23}$ [-]', r'$b_{33}$ [-]')
transparentBg = False


@timer
@jit
def readSlices(time, caseDir = '/media/yluan/Toshiba External Drive', caseName = 'ALM_N_H', fileNames = ('*',), skipCol = 3, skipRow = 0):
    caseFullPath = caseDir + '/' + caseName + '/Slices/' + str(time) + '/'
    fileNames = os.listdir(caseFullPath) if fileNames[0] in ('*', 'all') else fileNames

    slicesVal = {}
    slicesCoor = {}
    for fileName in fileNames:
        vals = np.genfromtxt(caseFullPath + fileName)
        # partition('.') removes anything after '.'
        slicesCoor[fileName.partition('.')[0]] = vals[skipRow:, :skipCol]
        slicesVal[fileName.partition('.')[0]] = vals[skipRow:, skipCol:]

    print('\n' + str(fileNames) + ' read')
    return slicesCoor, slicesVal


@timer
@jit
def interpolateVerticalSlices(x, y, z, vals, precisionX = 1500j, precisionY = 1500j, precisionZ = 500j, interpMethod = 'cubic'):
    # Known x and z coordinates, to be interpolated later
    pointsXZ = np.vstack((x, z)).T
    # Interpolate x and z according to precisions

    # x2D, z2D = np.mgrid[10:x.max():precisionX, z.min():z.max():precisionZ]
    x2D, z2D = np.mgrid[x.min():x.max():precisionX, z.min():z.max():precisionZ]

    # Then interpolate y in the same fashion of x
    y2D, _ = np.mgrid[y.min():y.max():precisionY, z.min():z.max():precisionZ]

    # Decompose the vector/tensor of slice values
    # If vector, order is x, y, z
    # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
    valsDecomp = {}
    for i in range(vals.shape[1]):
        # valsDecomp[str(i)] = vals[:, i]
        # Each component is interpolated from the known locations pointsXZ to refined fields (x2D, z2D)
        valsDecomp[str(i)] = griddata(pointsXZ, vals[:, i].ravel(), (x2D, z2D), method = interpMethod)

    return x2D, y2D, z2D, valsDecomp


"""
Read, Decompose and Plot Slices Data
"""
slicesCoor, slicesVal = readSlices(time = time, caseDir = caseDir, caseName = caseName, fileNames = fileNames)

for sliceName in sliceNames:
    x2D, y2D, z2D, valsDecomp = interpolateVerticalSlices(slicesCoor[sliceName][:, 0], slicesCoor[sliceName][:, 1], slicesCoor[sliceName][:, 2], slicesVal[sliceName], precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)

    # For anisotropic stress tensor bij
    # bij = Rij/(2k) - 1/3*deltaij
    # where Rij is uuPrime2, k = 1/2trace(Rij), deltaij is Kronecker delta
    if propertyName is 'uuPrime2':
        # k in the interpolated mesh
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        k = 0.5*(valsDecomp['0'] + valsDecomp['3'] + valsDecomp['5'])
        # Convert Rij to bij
        for key, val in valsDecomp.items():
            valsDecomp[key] = val/(2.*k) - 1/3. if key in ('0', '3', '5') else val/(2.*k)

    plotsLabel = iter(valLabels)
    for key, val in valsDecomp.items():
        slicePlot = Plot2D_InsetZoom(x2D, z2D, zoomBox = (1000, 2500, 0, 500), z2D = val, equalAxis = True, name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim, show = show, xLabel = xLabel, yLabel = zLabel, save = save, zLabel = next(plotsLabel))
        # slicePlot = PlotSurfaceSlices3D(x2D, y2D, z2D, val, name = sliceName + '_' + key + '_3d', figDir = figDir, xLim = xLim, yLim = yLim, zLim = zLim, show = show, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = next(plotsLabel))

        slicePlot.initializeFigure()
        slicePlot.plotFigure(contourLvl = 100)
        # slicePlot.plotFigure()
        slicePlot.finalizeFigure(transparentBg = transparentBg)




