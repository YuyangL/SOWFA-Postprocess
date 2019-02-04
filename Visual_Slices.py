"""
Read and Visualize Horizontal/Vertical Slices in 2/3D
"""
import numpy as np
import os
from numba import njit, jit
from Utilities import timer
from scipy.interpolate import griddata
from PlottingTool import Plot2D, Plot2D_InsetZoom, PlotSurfaceSlices3D, PlotContourSlices3D

caseDir = 'J:'  # '/media/yluan/Toshiba External Drive/'
caseName = 'ALM_N_H'  # 'ALM_N_H_ParTurb'
time = 22000.0558025  # 22000.0918025
# time = 20000.9038025
# sliceNames = ['alongWind', 'groundHeight', 'hubHeight', 'oneDaboveHubHeight', 'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# For Upwind and Downwind turbines
# sliceNames = ['oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# # For Parallel Turbines
# sliceNames = ['alongWindRotorOne', 'alongWindRotorTwo', 'twoDupstreamTurbines', 'rotorPlane', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines', 'sixDdownstreamTurbines']
# sliceNames = ['groundHeight', 'hubHeight', 'oneDaboveHubHeight']
sliceNames = ['rotorPlaneTwo', 'rotorPlaneOne']
sliceDir = 'vertical'
sliceOffsets = (5, 90, 216)
propertyName = 'U'
fileExt = '.raw'
# Combine propertyName with sliceNames and Subscript to form the full file names
# Don't know why I had to copy it...
fileNames = sliceNames.copy()
for i, name in enumerate(sliceNames):
    sliceNames[i] = propertyName + '_' + name + '_Slice'
    fileNames[i] = sliceNames[i] + fileExt

precisionX, precisionY, precisionZ = 1500j, 1500j, 500j
interpMethod = 'cubic'

"""
Plot Settings
"""
figDir = caseDir + caseName + '/Slices/Result/' + str(time)
try:
    os.makedirs(figDir)
except FileExistsError:
    pass

figWidth = 'full'
# View angle best (15, -40) for rotor plane
viewAngle, equalAxis = (15, -40), True
xLim, yLim, zLim = (0, 3000), (0, 3000), (0, 1000)
show, save = False, True
xLabel, yLabel, zLabel = r'$x$ [m]', r'$y$ [m]', r'$z$ [m]'
# valLabels = (r'$b_{11}$ [-]', r'$b_{12}$ [-]', r'$b_{13}$ [-]', r'$b_{22}$ [-]', r'$b_{23}$ [-]', r'$b_{33}$ [-]')
# valLabels = (r'$\langle u\rangle$ [m/s]', r'$\langle v\rangle$ [m/s]', r'$\langle w\rangle$ [m/s]')
valLabels = (r'$U$ [m/s]', r'$U$ [m/s]', r'$U$ [m/s]')
transparentBg = False


@timer
@jit
def readSlices(time, caseDir = '/media/yluan/Toshiba External Drive', caseName = 'ALM_N_H', fileNames = ('*',), skipCol = 3, skipRow = 0):
    caseFullPath = caseDir + '/' + caseName + '/Slices/' + str(time) + '/'
    fileNames = os.listdir(caseFullPath) if fileNames[0] in ('*', 'all') else fileNames

    slicesVal, slicesDir, slicesCoor = {}, {}, {}
    for fileName in fileNames:
        vals = np.genfromtxt(caseFullPath + fileName)
        # partition('.') removes anything after '.'
        slicesCoor[fileName.partition('.')[0]] = vals[skipRow:, :skipCol]

        # NOT WORKING!
        slicesDir[fileName.partition('.')[0]] = 'vertical' if abs((vals[skipRow:, skipCol]).max() - (vals[skipRow:, skipCol]).min()) < 0.1*((vals[skipRow:, skipCol]).max()) else 'horizontal'
        slicesVal[fileName.partition('.')[0]] = vals[skipRow:, skipCol:]

    print('\n' + str(fileNames) + ' read')
    return slicesCoor, slicesDir, slicesVal


@timer
@jit
def interpolateSlices(x, y, z, vals, sliceDir = 'vertical', precisionX = 1500j, precisionY = 1500j, precisionZ = 500j, interpMethod = 'cubic'):
    # Bound the coordinates to be interpolated in case data wasn't available in those borders
    bnd = (1.0001, 0.9999)
    if sliceDir is 'vertical':
        # Known x and z coordinates, to be interpolated later
        knownPoints = np.vstack((x, z)).T
        # Interpolate x and z according to precisions
        x2D, z2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]

        # Then interpolate y in the same fashion of x
        y2D, _ = np.mgrid[y.min()*bnd[0]:y.max()*bnd[1]:precisionY, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
        # In case the vertical slice is at a negative angle,
        # i.e. when x goes from low to high, y goes from high to low,
        # flip y2D from low to high to high to low
        y2D = np.flipud(y2D) if x[0] > x[1] else y2D
    else:
        knownPoints = np.vstack((x, y)).T
        x2D, y2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, y.min()*bnd[0]:y.max()*bnd[1]:precisionY]
        _, z2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]

    # Decompose the vector/tensor of slice values
    # If vector, order is x, y, z
    # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
    valsDecomp = {}
    for i in range(vals.shape[1]):
        if sliceDir is 'vertical':
            # Each component is interpolated from the known locations pointsXZ to refined fields (x2D, z2D)
            valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2D, z2D), method = interpMethod)
        else:
            valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2D, y2D), method = interpMethod)

    return x2D, y2D, z2D, valsDecomp


@timer
@jit
def calculateAnisotropicTensor(valsDecomp):
    # k in the interpolated mesh
    # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
    k = 0.5*(valsDecomp['0'] + valsDecomp['3'] + valsDecomp['5'])
    # Convert Rij to bij
    for key, val in valsDecomp.items():
        valsDecomp[key] = val/(2.*k) - 1/3. if key in ('0', '3', '5') else val/(2.*k)

    return valsDecomp

@timer
@jit
def mergeHorizontalComponent(valsDecomp):
    valsDecomp['hor'] = np.sqrt(valsDecomp['0']**2 + valsDecomp['1']**2)
    return valsDecomp


"""
Read, Decompose and Plot Slices Data
"""
slicesCoor, slicesDir, slicesVal = readSlices(time = time, caseDir = caseDir, caseName = caseName, fileNames = fileNames)

# Initialize slice lists for multple slice plots in one 3D figure
horSliceLst, zSliceLst, listX2D, listY2D, listZ2D = [], [], [], [], []
# Go through slices
for sliceName in sliceNames:
    x2D, y2D, z2D, valsDecomp = interpolateSlices(slicesCoor[sliceName][:, 0], slicesCoor[sliceName][:, 1], slicesCoor[sliceName][:, 2], slicesVal[sliceName], sliceDir = sliceDir, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)

    # xLim, yLim, zLim = (x2D.min(), x2D.max()), (y2D.min(), y2D.max()), (z2D.min(), z2D.max())

    # For anisotropic stress tensor bij
    # bij = Rij/(2k) - 1/3*deltaij
    # where Rij is uuPrime2, k = 1/2trace(Rij), deltaij is Kronecker delta
    if propertyName is 'uuPrime2':
        valsDecomp = calculateAnisotropicTensor(valsDecomp)
    elif propertyName is 'U':
        valsDecomp = mergeHorizontalComponent(valsDecomp)

    plotsLabel = iter(valLabels)
    # for key, val in valsDecomp.items():
        # if slicesDir[sliceName] is 'vertical':
        #     slicePlot = Plot2D(x2D, z2D, z2D = val, equalAxis = True,
        #                                  name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim,
        #                                  show = show, xLabel = xLabel, yLabel = zLabel, save = save,
        #                                  zLabel = next(plotsLabel))
        #
        # else:
            # slicePlot = Plot2D(x2D, y2D, z2D = val, equalAxis = True,
            #                    name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = yLim,
            #                    show = show, xLabel = xLabel, yLabel = yLabel, save = save,
            #                    zLabel = next(plotsLabel))
        # slicePlot = Plot2D_InsetZoom(x2D, z2D, zoomBox = (1000, 2500, 0, 500), z2D = val, equalAxis = True, name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim, show = show, xLabel = xLabel, yLabel = zLabel, save = save, zLabel = next(plotsLabel))
        # slicePlot = PlotSurfaceSlices3D(x2D, y2D, z2D, val, name = sliceName + '_' + key + '_3d', figDir = figDir, xLim = xLim, yLim = yLim, zLim = zLim, show = show, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = next(plotsLabel))

        # slicePlot.initializeFigure()
        # slicePlot.plotFigure(contourLvl = 100)
        # # slicePlot.plotFigure()
        # slicePlot.finalizeFigure(transparentBg = transparentBg, cbarOrientate = 'vertical')


    horSliceLst.append(valsDecomp['hor'])
    zSliceLst.append(valsDecomp['2'])
    listX2D.append(x2D)
    listY2D.append(y2D)
    listZ2D.append(z2D)

if sliceDir is 'horizontal':
    slicePlot = PlotContourSlices3D(x2D, y2D, horSliceLst, sliceOffsets = sliceOffsets, contourLvl = 100, zLim = (0, 216), gradientBg = False, name = str(propertyName) + 'slices_hor', figDir = figDir, show = show, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, cmapLabel = r'$U_{\rm{hor}}$ [m/s]', save = save, cbarOrientate = 'vertical')
else:
    slicePlot = PlotSurfaceSlices3D(listX2D, listY2D, listZ2D, horSliceLst, name = str(propertyName) + 'hor_RotorPlane_Slices', figDir = figDir,
                                    show = show, xLabel = xLabel,
                                    yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = r'$U_{\rm{hor}}$ [m/s]', viewAngles = viewAngle, figWidth = figWidth, xLim = xLim, yLim = yLim, zLim = zLim, equalAxis = equalAxis)

slicePlot.initializeFigure()
slicePlot.plotFigure()
# slicePlot.plotFigure()
slicePlot.finalizeFigure(transparentBg = transparentBg)


# if sliceDir is 'horizontal':
#     slicePlot = PlotContourSlices3D(x2D, y2D, zSliceLst, sliceOffsets = sliceOffsets, contourLvl = 100,
#                                     xLim = (0, 3000), yLim = (0, 3000), zLim = (0, 216), gradientBg = False,
#                                     name = str(propertyName) + 'slices_z', figDir = figDir, show = show,
#                                     xLabel = xLabel, yLabel = yLabel, zLabel = zLabel,
#                                     cmapLabel = r'$U_{z}$ [m/s]', save = save, cbarOrientate = 'vertical')
# else:
#     slicePlot = PlotSurfaceSlices3D(listX2D, listY2D, listZ2D, zSliceLst,
#                                     name = str(propertyName) + 'z_RotorPlane_Slices', figDir = figDir, show = show, xLabel = xLabel,
#                                     yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = r'$U_{z}$ [m/s]', viewAngles = viewAngle, figWidth = figWidth, xLim = xLim, yLim = yLim, zLim = zLim)
#
# slicePlot.initializeFigure()
# slicePlot.plotFigure()
# # slicePlot.plotFigure()
# slicePlot.finalizeFigure(transparentBg = transparentBg)




