"""
Read and Visualize Horizontal/Vertical Slices in 2/3D
"""
import numpy as np
import os
from numba import njit, jit
from Utilities import timer
from scipy.interpolate import griddata
from PlottingTool import Plot2D, Plot2D_InsetZoom, PlotSurfaceSlices3D, PlotContourSlices3D
try:
    import PostProcess_AnisotropyTensor as PPAT
except ImportError:
    raise ImportError('\nNo module named PostProcess_AnisotropyTensor. Check setup.py and run'
                      '\npython setup.py build_ext --inplace')

try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
from matplotlib.path import Path
import PostProcess_SliceData as PPSD

"""
User Inputs
"""
time = '23243.2156219'  # 'latest'
caseDir = '/media/yluan'
caseName = 'URANS/N_H_OneTurb'  # 'ALM_N_H_ParTurb'
# propertyNames = ('kResolved', 'kSGSmean')
propertyNames = ('U',)
# sliceNames = ('rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'sevenDdownstreamTurbine')
sliceNames = ('rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine')
# sliceNames = ('hubHeight', 'quarterDaboveHub', 'turbineApex')
# sliceNames = ('groundHeight', 'hubHeight', 'turbineApex')
# Subscript for the slice names
sliceNamesSub = 'Slice'
# Height of the horizontal slices, only used for 3D horizontal slices plot
horSliceOffsets = (90, 121.5, 153)
resultFolder = 'Result'
# Confine the region of interest, list grows with number of slices
# (800, 1800, 800, 1800, 0, 405) good for rotor plane slice
# (600, 2000, 600, 2000, 0, 405) good for hubHeight slice
# (800, 2200, 800, 2200, 0, 405) good for along wind slice
# Overriden in Process User Inputs
confineBox = ((800, 1800, 800, 1800, 0, 405),)  # (None,), ((xmin, xmax, ymin, ymax, zmin, zmax),)
# Orientation of x-axis in x-y plane, in case of angled flow direction
# Only used for values decomposition and confineBox
# Angle in rad and counter-clockwise
xOrientate = 6/np.pi
# Turbine radius, only used for confineBox
r = 63
# For calculating total <epsilon> only
nu = 1e-5


"""
Plot Settings
"""
# Which type(s) of plot to make
plotType = '3D'  # '2D', '3D', 'all'
# Total number cells intended to plot via interpolation
targetCells = 1e6
interpMethod = 'cubic'
# Number of contours, only for 2D plots or 3D horizontal slice plots
contourLvl = 100
# Label of the property, could be overridden below
valLabel = 'Data'
ext = 'png'
show, save = False, True
dpi = 600


"""
Process User Inputs
"""
# Ensure sliceNames is a tuple
sliceNames = (sliceNames,) if isinstance(sliceNames, str) else sliceNames
# Confined region auto definition
if caseName in ('ALM_N_H_OneTurb', 'URANS/N_H_OneTurb'):
    # For rotor plane vertical slices
    if sliceNames[0] == 'rotorPlane':
        confineBox = ((1023.583 + r/2, 1212.583 - r/2, 1115.821 + r/2,
                       1443.179 - r/2,
                       0,
                       216 - r/2),
                      (1132.702, 1321.702, 1178.821, 1506.179, 0, 216),
                      (1350.94, 1539.94, 1304.821, 1632.179, 0, 216),
                      (1569.179, 1758.179, 1430.821, 1758.179, 0, 216))

    # For horizontal slices
    elif sliceNames[0] in ('groundHeight', 'hubHeight'):
        # Confinement for z doesn't matter since the slices are horizontal
        confineBox = ((600, 2000, 600, 2000, 0, 216),)*len(sliceNames)

elif caseName == 'ALM_N_H_ParTurb':
    if sliceNames[0] == 'rotorPlane':
        confineBox = ((1023.583 + r/2, 1212.583 - r/2, 1115.821 + r/2,
                       1443.179 - r/2,
                       0,
                       216 - r/2),
                      (1132.702, 1321.702, 1178.821, 1506.179, 0, 216),
                      (1350.94, 1539.94, 1304.821, 1632.179, 0, 216),
                      (1569.179, 1758.179, 1430.821, 1758.179, 0, 216))
    elif sliceNames[0] in ('groundHeight', 'hubHeight'):
        # Confinement for z doesn't matter since the slices are horizontal
        confineBox = ((700, 2300, 700, 2300, 0, 216),)*len(sliceNames)

# Automatic viewAngle and figure settings, only for 3D plots
if sliceNames[0] == 'rotorPlane':
    viewAngle = (20, -100)
    equalAxis, figSize, figWidth = True, (1, 1), 'full'
elif sliceNames[0] in ('groundHeight', 'hubHeight'):
    viewAngle = (20, -115)
    equalAxis, figSize, figWidth = False, (2, 1), 'half'
else:
    viewAngle = (20, -100)
    equalAxis, figSize, figWidth = True, (1, 1), 'full'

# Unify plotType user inputs
if plotType in ('2D', '2d'):
    plotType = '2D'
elif plotType in ('3D', '3d'):
    plotType = '3D'
elif plotType in ('all', 'All', '*'):
    plotType = 'all'

if 'kResolved' in propertyNames and 'kSGSmean' in propertyNames:
    valLabel = r'$\langle k\rangle$ [m$^2$/s$^2$]'
elif 'epsilonSGSmean' in propertyNames and 'nuSGSmean' in propertyNames:
    valLabel = r'$\langle \epsilon \rangle$ [m$^2$/s$^3$]'
elif 'gradUAvg' in propertyNames:
    valLabel = (r'$\nabla \textrm{U}$ [1/s]')


"""
Read Slice Data
"""
# Initialize case
case = PPSD.SliceProperties(time = time, caseDir = caseDir, caseName = caseName, xOrientate = xOrientate, resultFolder =
resultFolder)
# Read slices
case.readSlices(propertyNames = propertyNames, sliceNames = sliceNames, sliceNamesSub = sliceNamesSub)

listX2D, listY2D, listZ2D, listVals3D = [], [], [], []
# Go through specified slices and flow properties
for i in range(len(case.sliceNames)):
    # for i, sliceName in enumerate(case.sliceNames):
    sliceName = case.sliceNames[i]
    vals2D = case.slicesVal[sliceName]
    # If kResolved and kSGSmean in propertyNames, get total kMean
    if 'kResolved' in propertyNames and 'kSGSmean' in propertyNames:
        print(' Calculating total <k> for {}...'.format(sliceNames[i]))
        sliceName2 = case.sliceNames[i + len(sliceNames)]
        vals2D += case.slicesVal[sliceName2]
    # Else if epsilonSGSmean and nuSGSmean in propertyNames then get total epsilonMean
    # By assuming isotropic homogeneous turbulence and
    # <epsilon> = <epsilonSGS>/(1 - 1/(1 + <nuSGS>/nu))
    elif 'epsilonSGSmean' in propertyNames and 'nuSGSmean' in propertyNames:
        print(' Calculating total <epsilon> for {}...'.format(sliceNames[i]))
        sliceName2 = case.sliceNames[i + len(sliceNames)]
        vals2D_2 = case.slicesVal[sliceName2]
        # Determine which vals2D is epsilonSGSmean or nuSGSmean
        nuSGSmean, epsilonSGSmean = (vals2D, vals2D_2) if 'epsilonSGSmean' in sliceName2 else (vals2D_2, vals2D)
        # Calculate epsilonMean
        vals2D = case.calcSliceMeanDissipationRate(epsilonSGSmean = epsilonSGSmean, nuSGSmean = nuSGSmean, nu = nu)

    # Interpolation
    x2D, y2D, z2D, vals3D = case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], vals2D, sliceOrientate =
    case.slicesOrientate[sliceName], xOrientate = case.xOrientate,
                                                                     targetCells = targetCells,
                                                                     interpMethod = interpMethod,
                                                                     confineBox = confineBox[i])

    # Flatten if vals3D only have one component like a scalar field
    if vals3D.shape[2] == 1:
        vals3D = vals3D.reshape((vals3D.shape[0], vals3D.shape[1]))

    # Calculate magnitude if U
    if 'U' in propertyNames or 'UAvg' in propertyNames:
        vals3D = np.sqrt(vals3D[:, :, 0]**2 + vals3D[:, :, 1]**2 + vals3D[:, :, 2]**2)

    # Append 2D mesh to a list for 3D plots
    if plotType in ('3D', 'all'):
        listX2D.append(x2D)
        listY2D.append(y2D)
        listZ2D.append(z2D)
        listVals3D.append(vals3D)

    # Determine the unit along the vertical slice since it's angled, only for 2D plots of vertical slices
    if case.slicesOrientate[sliceName] == 'vertical':
        # # If angle from x-axis is 45 deg or less
        # if lx >= ly:
        #     xOrientate = np.arctan(lx/ly)
        if confineBox[0] is None:
            lx = np.max(x2D) - np.min(x2D)
            ly = np.max(y2D) - np.min(y2D)
        else:
            lx = confineBox[i][1] - confineBox[i][0]
            ly = confineBox[i][3] - confineBox[i][2]

        r2D = np.linspace(0, np.sqrt(lx**2 + ly**2), x2D.shape[0])

    # Break if i finishes all kResolved or kSGSmean
    if 'kResolved' in propertyNames and 'kSGSmean' in propertyNames and i == (len(sliceNames) - 1):
        break
    elif 'epsilonSGSmean' in propertyNames and 'nuSGSmean' in propertyNames and i == (len(sliceNames) - 1):
        break


    """
    Plotting
    """
    xLabel, yLabel = (r'$r$ [m]', r'$z$ [m]') \
        if case.slicesOrientate[sliceName] == 'vertical' else \
        (r'$x$ [m]', r'$y$ [m]')
    # Figure name
    if 'kResolved' in propertyNames and 'kSGSmean' in propertyNames:
        figName = 'kMean_' + sliceNames[i] + sliceNamesSub
    elif 'epsilonSGSmean' in propertyNames and 'nuSGSmean' in propertyNames:
        figName = 'epsilonMean_' + sliceNames[i] + sliceNamesSub
    else:
        figName = sliceName

    if plotType in ('2D', 'all'):
        slicePlot = Plot2D(x2D, y2D, vals3D, name = figName, xLabel = xLabel, yLabel = yLabel, zLabel = valLabel, save = save, show = show, figDir = case.resultPath)
        slicePlot.initializeFigure()
        slicePlot.plotFigure(contourLvl = contourLvl)
        slicePlot.finalizeFigure()

if plotType in ('3D', 'all'):
    zLabel = r'$z$ [m]'
    # Figure name for 3D plots
    if 'kResolved' in propertyNames and 'kSGSmean' in propertyNames:
        figName3D = 'kMean_' + str(sliceNames)
    elif 'epsilonSGSmean' in propertyNames and 'nuSGSmean' in propertyNames:
        figName3D = 'epsilonMean_' + str(sliceNames)
    else:
        figName3D = str(case.sliceNames)

    if case.slicesOrientate[sliceName] == 'horizontal':
        slicePlot3D = PlotContourSlices3D(listX2D, listY2D, listVals3D, horSliceOffsets, contourLvl = contourLvl, gradientBg = False, name = figName3D, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, cmapLabel = valLabel, save = save, show = show, figDir = case.resultPath, viewAngles = viewAngle, figWidth = figWidth, equalAxis = equalAxis, cbarOrientate = 'vertical')
    elif case.slicesOrientate[sliceName] == 'vertical':
        slicePlot3D = PlotSurfaceSlices3D(listX2D, listY2D, listZ2D, listVals3D, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, cmapLabel = valLabel, name = str(sliceNames), save = save, show = show, figDir = case.resultPath, viewAngles = viewAngle, figWidth = figWidth, equalAxis = equalAxis, cbarOrientate = 'horizontal')

    slicePlot3D.initializeFigure(figSize = figSize)
    slicePlot3D.plotFigure()
    slicePlot3D.finalizeFigure()








# """
# User Inputs
# """
# caseDir = 'J:'  # '/media/yluan/Toshiba External Drive/'
# caseDir = '/media/yluan/Toshiba External Drive/'
# caseName = 'ALM_N_H_OneTurb'  # 'ALM_N_H_ParTurb'
# time = 23275.1388025  # 22000.0918025 20000.9038025
# # sliceNames = ['alongWind', 'groundHeight', 'hubHeight', 'oneDaboveHubHeight', 'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# # For Upwind and Downwind turbines
# # sliceNames = ['oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# # # For Parallel Turbines
# # sliceNames = ['alongWindRotorOne', 'alongWindRotorTwo', 'twoDupstreamTurbines', 'rotorPlane', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines', 'sixDdownstreamTurbines']
# # sliceNames = ['groundHeight', 'hubHeight', 'oneDaboveHubHeight']
# # sliceNames = ['rotorPlane','sixDdownstreamTurbines']
# sliceNames = ['alongWind']
# # Only for PlotContourSlices3D
# sliceOffsets = (5, 90, 153)
# propertyName = 'uuPrime2'
# fileExt = '.raw'
# precisionX, precisionY, precisionZ = 1000j, 1000j, 333j
# interpMethod = 'nearest'
#
#
# """
# Plot Settings
# """
# figWidth = 'full'
# # View angle best (15, -40) for vertical slices in rotor plane
# viewAngle, equalAxis = (15, -45), True
# xLim, yLim, zLim = (0, 3000), (0, 3000), (0, 1000)
# show, save = False, True
# xLabel, yLabel, zLabel = r'$x$ [m]', r'$y$ [m]', r'$z$ [m]'
# # valLabels = (r'$b_{11}$ [-]', r'$b_{12}$ [-]', r'$b_{13}$ [-]', r'$b_{22}$ [-]', r'$b_{23}$ [-]', r'$b_{33}$ [-]')
# # valLabels = (r'$\langle u\rangle$ [m/s]', r'$\langle v\rangle$ [m/s]', r'$\langle w\rangle$ [m/s]')
# if propertyName == 'U':
#     valLabels = (r'$U$ [m/s]', r'$U$ [m/s]', r'$U$ [m/s]')
# elif propertyName == 'uuPrime2':
#     valLabels = (r'$b_{11}$ [-]', r'$b_{12}$ [-]', r'$b_{13}$ [-]', r'$b_{22}$ [-]', r'$b_{23}$ [-]', r'$b_{33}$ [-]', r'$k_{\rm{resolved}}$ [m$^2$/s$^2$]')
#
#
# """
# Process User Inputs
# """
# # Combine propertyName with sliceNames and Subscript to form the full file names
# # Don't know why I had to copy it...
# fileNames = sliceNames.copy()
# for i, name in enumerate(sliceNames):
#     sliceNames[i] = propertyName + '_' + name + '_Slice'
#     fileNames[i] = sliceNames[i] + fileExt
#
# figDir = caseDir + caseName + '/Slices/Result/' + str(time)
# try:
#     os.makedirs(figDir)
# except FileExistsError:
#     pass
#
#
# """
# Functions
# """
# @timer
# @jit
# def readSlices(time, caseDir = '/media/yluan/Toshiba External Drive', caseName = 'ALM_N_H', fileNames = ('*',), skipCol = 3, skipRow = 0):
#     caseFullPath = caseDir + '/' + caseName + '/Slices/' + str(time) + '/'
#     fileNames = os.listdir(caseFullPath) if fileNames[0] in ('*', 'all') else fileNames
#     slicesVal, slicesDir, slicesCoor = {}, {}, {}
#     for fileName in fileNames:
#         vals = np.genfromtxt(caseFullPath + fileName)
#         # partition('.') removes anything after '.'
#         slicesCoor[fileName.partition('.')[0]] = vals[skipRow:, :skipCol]
#         # If max(z) - min(z) < 1 then it's assumed horizontal
#         slicesDir[fileName.partition('.')[0]] = 'vertical' if (vals[skipRow:, skipCol - 1]).max() - (vals[skipRow:, skipCol - 1]).min() > 1. else 'horizontal'
#         slicesVal[fileName.partition('.')[0]] = vals[skipRow:, skipCol:]
#
#     print('\n' + str(fileNames) + ' read')
#     return slicesCoor, slicesDir, slicesVal
#
#
# @timer
# @jit
# def interpolateSlices(x, y, z, vals, sliceDir = 'vertical', precisionX = 1500j, precisionY = 1500j, precisionZ = 500j, interpMethod = 'cubic'):
#     # Bound the coordinates to be interpolated in case data wasn't available in those borders
#     bnd = (1.00001, 0.99999)
#     if sliceDir is 'vertical':
#         # Known x and z coordinates, to be interpolated later
#         knownPoints = np.vstack((x, z)).T
#         # Interpolate x and z according to precisions
#         x2D, z2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
#         # Then interpolate y in the same fashion of x
#         y2D, _ = np.mgrid[y.min()*bnd[0]:y.max()*bnd[1]:precisionY, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
#         # In case the vertical slice is at a negative angle,
#         # i.e. when x goes from low to high, y goes from high to low,
#         # flip y2D from low to high to high to low
#         y2D = np.flipud(y2D) if x[0] > x[1] else y2D
#     else:
#         knownPoints = np.vstack((x, y)).T
#         x2D, y2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, y.min()*bnd[0]:y.max()*bnd[1]:precisionY]
#         _, z2D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
#
#     # Decompose the vector/tensor of slice values
#     # If vector, order is x, y, z
#     # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
#     valsDecomp = {}
#     for i in range(vals.shape[1]):
#         if sliceDir is 'vertical':
#             # Each component is interpolated from the known locations pointsXZ to refined fields (x2D, z2D)
#             valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2D, z2D), method = interpMethod)
#         else:
#             valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2D, y2D), method = interpMethod)
#
#     return x2D, y2D, z2D, valsDecomp
#
#
# @timer
# @jit
# def calculateAnisotropicTensor(valsDecomp):
#     # k in the interpolated mesh
#     # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
#     k = 0.5*(valsDecomp['0'] + valsDecomp['3'] + valsDecomp['5'])
#     # Convert Rij to bij
#     for key, val in valsDecomp.items():
#         valsDecomp[key] = val/(2.*k) - 1/3. if key in ('0', '3', '5') else val/(2.*k)
#
#     return valsDecomp, k
#
#
# @timer
# @jit
# def mergeHorizontalComponent(valsDecomp):
#     valsDecomp['hor'] = np.sqrt(valsDecomp['0']**2 + valsDecomp['1']**2)
#     return valsDecomp
#
#
# """
# Read, Decompose and Plot 2/3D Slices
# """
# slicesCoor, slicesDir, slicesVal = readSlices(time = time, caseDir = caseDir, caseName = caseName, fileNames = fileNames)
#
# # Initialize slice lists for multple slice plots in one 3D figure
# horSliceLst, zSliceLst, listX2D, listY2D, listZ2D = [], [], [], [], []
# # Go through slices
# for sliceName in sliceNames:
#     x2D, y2D, z2D, valsDecomp = interpolateSlices(slicesCoor[sliceName][:, 0], slicesCoor[sliceName][:, 1], slicesCoor[sliceName][:, 2], slicesVal[sliceName], sliceDir = slicesDir[sliceName], precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)
#
#     # For anisotropic stress tensor bij
#     # bij = Rij/(2k) - 1/3*deltaij
#     # where Rij is uuPrime2, k = 1/2trace(Rij), deltaij is Kronecker delta
#     if propertyName == 'uuPrime2':
#         valsDecomp, k = calculateAnisotropicTensor(valsDecomp)
#         valsDecomp['kResolved'] = k
#     elif propertyName == 'U':
#         valsDecomp = mergeHorizontalComponent(valsDecomp)
#
#
#     """
#     2D Contourf Plots
#     """
#     xLim, yLim, zLim = (x2D.min(), x2D.max()), (y2D.min(), y2D.max()), (z2D.min(), z2D.max())
#     plotsLabel = iter(valLabels)
#     for key, val in valsDecomp.items():
#         # if slicesDir[sliceName] is 'vertical':
#         #     slicePlot = Plot2D(x2D, z2D, z2D = val, equalAxis = True,
#         #                                  name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim,
#         #                                  show = show, xLabel = xLabel, yLabel = zLabel, save = save,
#         #                                  zLabel = next(plotsLabel))
#         #
#         # else:
#         #     slicePlot = Plot2D(x2D, y2D, z2D = val, equalAxis = True,
#         #                        name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = yLim,
#         #                        show = show, xLabel = xLabel, yLabel = yLabel, save = save,
#         #                        zLabel = next(plotsLabel))
#         # slicePlot = Plot2D_InsetZoom(x2D, z2D, zoomBox = (1000, 2500, 0, 500), z2D = val, equalAxis = True, name = sliceName + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim, show = show, xLabel = xLabel, yLabel = zLabel, save = save, zLabel = next(plotsLabel))
#         # plotType = 'contour2D'
#
#         slicePlot = PlotSurfaceSlices3D(x2D, y2D, z2D, val, name = sliceName + '_' + key + '_3d', figDir = figDir, xLim = xLim, yLim = yLim, zLim = zLim, show = show, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = next(plotsLabel), viewAngles = viewAngle, figWidth = figWidth)
#         plotType = 'surface3D'
#
#         slicePlot.initializeFigure()
#         if plotType == 'contour2D':
#             slicePlot.plotFigure(contourLvl = 100)
#         else:
#             slicePlot.plotFigure()
#
#         slicePlot.finalizeFigure()
#
#     if propertyName == 'U':
#         horSliceLst.append(valsDecomp['hor'])
#         zSliceLst.append(valsDecomp['2'])
#
#     listX2D.append(x2D)
#     listY2D.append(y2D)
#     listZ2D.append(z2D)


"""
Multiple Slices of Horizontal Component 3D Plot
"""
# if slicesDir[sliceName] is 'horizontal':
#     slicePlot = PlotContourSlices3D(x2D, y2D, horSliceLst, sliceOffsets = sliceOffsets, contourLvl = 100, zLim = (0, 216), gradientBg = False, name = str(sliceNames) + '_hor', figDir = figDir, show = show, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, cmapLabel = r'$U_{\rm{hor}}$ [m/s]', save = save, cbarOrientate = 'vertical')
# else:
#     slicePlot = PlotSurfaceSlices3D(listX2D, listY2D, listZ2D, horSliceLst, name = str(sliceNames) + '_hor', figDir = figDir, show = show, xLabel = xLabel,
#                                     yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = r'$U_{\rm{hor}}$ [m/s]', viewAngles = viewAngle, figWidth = figWidth, xLim = xLim, yLim = yLim, zLim = zLim, equalAxis = equalAxis)
#
# slicePlot.initializeFigure()
# slicePlot.plotFigure()
# slicePlot.finalizeFigure()


"""
Multiple Slices of Z Component 3D Plot
"""
# if slicesDir[sliceName] is 'horizontal':
#     slicePlot = PlotContourSlices3D(x2D, y2D, zSliceLst, sliceOffsets = sliceOffsets, contourLvl = 100,
#                                     xLim = (0, 3000), yLim = (0, 3000), zLim = (0, 216), gradientBg = False,
#                                     name = str(sliceNames) + '_z', figDir = figDir, show = show,
#                                     xLabel = xLabel, yLabel = yLabel, zLabel = zLabel,
#                                     cmapLabel = r'$U_{z}$ [m/s]', save = save, cbarOrientate = 'vertical')
# else:
#     slicePlot = PlotSurfaceSlices3D(listX2D, listY2D, listZ2D, zSliceLst,
#                                     name = str(sliceNames) + '_z', figDir = figDir, show = show, xLabel = xLabel,
#                                     yLabel = yLabel, zLabel = zLabel, save = save, cmapLabel = r'$U_{z}$ [m/s]', viewAngles = viewAngle, figWidth = figWidth, xLim = xLim, yLim = yLim, zLim = zLim, equalAxis = equalAxis)
#
# slicePlot.initializeFigure()
# slicePlot.plotFigure()
# slicePlot.finalizeFigure()




