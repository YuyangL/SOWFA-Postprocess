import time as t
from PlottingTool import PlotImageSlices3D, BaseFigure, Plot2D_InsetZoom, PlotContourSlices3D
from PostProcess_SliceData import *
try:
    import PostProcess_AnisotropyTensor as PPAT
except ImportError:
    raise ImportError('\nNo module named PostProcess_AnisotropyTensor. Check setup.py and run'
                      '\npython setup.py build_ext --inplace')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy import ndimage
import pickle
import numpy as np
# import visvis as vv

"""
User Inputs
"""
time = 'latest'
caseDir = 'J:'
caseDir = '/media/yluan/1'
caseName = 'ALM_N_H_OneTurb'
propertyName = 'uuPrime2'
# sliceNames = ('rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'sevenDdownstreamTurbine')
sliceNames = ('groundHeight', 'hubHeight', 'turbineApexHeight')
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
dumpResult = False


"""
Plot Settings
"""
# Which type(s) of plot to make
plotType = '3D'  # '2D', '3D', 'all'
# Total number cells intended to plot via interpolation
targetCells = 1e6
# precisionX, precisionY, precisionZ = 3000, 3000, 1000
interpMethod = 'cubic'
showBaryExample = False
c_offset, c_exp = 0.65, 5.
baryMapExampleName = 'barycentric_colormap'
ext = 'png'
show, save = False, True
dpi = 1000


"""
Process User Inputs
"""
# For rotor plane slices
if caseName == 'ALM_N_H_OneTurb':
    if sliceNames[0] == 'rotorPlane':
        confineBox = ((1023.583 + r/2, 1212.583 - r/2, 1115.821 + r/2,
                       1443.179 - r/2,
                       0,
                       216 - r/2),
                      (1132.702, 1321.702, 1178.821, 1506.179, 0, 216),
                      (1350.94, 1539.94, 1304.821, 1632.179, 0, 216),
                      (1569.179, 1758.179, 1430.821, 1758.179, 0, 216))
        viewAngle = (20, -100)
        equalAxis, figSize, figWidth = True, (1, 1), 'full'
    elif sliceNames[0] == 'groundHeight':
        # Confinement for z doesn't matter since the slices are horizontal
        confineBox = ((600, 2000, 600, 2000, 0, 216),)*len(sliceNames)
        viewAngle = (25, -75)
        equalAxis, figSize, figWidth = False, (2, 1), 'half'
else:
    viewAngle = (20, -100)
    equalAxis, figSize, figWidth = True, (1, 1), 'full'

if plotType in ('2D', '2d'):
    plotType = '2D'
elif plotType in ('3D', '3d'):
    plotType = '3D'
elif plotType in ('all', 'All', '*'):
    plotType = 'all'


"""
Read Slice Data
"""
case = SliceProperties(time = time, caseDir = caseDir, caseName = caseName, xOrientate = xOrientate, resultFolder =
resultFolder)
case.readSlices(propertyName = propertyName, sliceNames = sliceNames)
listX2D, listY2D, listZ2D, listRGB = [], [], [], []
# Go through specified slices
for i, sliceName in enumerate(case.sliceNames):
    """
    Process Uninterpolated Anisotropy Tensor
    """
    vals2D = case.slicesVal[sliceName]
    # x2D, y2D, z2D, vals3D = \
    #     case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], case.slicesVal[sliceName], sliceOrientate = case.slicesOrientate[sliceName], xOrientate = case.xOrientate, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)

    # Another implementation of processAnisotropyTensor() in Cython
    t0 = t.time()
    # vals3D, tensors, eigValsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor(vals3D)
    # In each eigenvector 3 x 3 matrix, 1 col is a vector
    vals2D, tensors, eigValsGrid, eigVecsGrid = PPAT.processAnisotropyTensor_Uninterpolated(vals2D, realizeIter = 0)
    t1 = t.time()
    print('\nFinished processAnisotropyTensor_Uninterpolated in {:.4f} s'.format(t1 - t0))

    # valsDecomp, tensors, eigValsGrid = case.processAnisotropyTensor_Fast(valsDecomp)
    # vals2D, tensors, eigValsGrid = case.processAnisotropyTensor_Uninterpolated(vals2D)

    xBary, yBary, rgbVals = case.getBarycentricMapCoordinates(eigValsGrid, c_offset = c_offset, c_exp = c_exp)


    """
    Interpolation
    """
    x2D, y2D, z2D, rgbVals = \
        case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], rgbVals, sliceOrientate =
                                                 case.slicesOrientate[sliceName], xOrientate = case.xOrientate,
                                                 targetCells = targetCells,
                                                 interpMethod = interpMethod,
                                                 confineBox = confineBox[i])
    # Append 2D mesh to a list for 3D plots
    if plotType in ('3D', '3d', 'all', 'All'):
        listX2D.append(x2D)
        listY2D.append(y2D)
        listZ2D.append(z2D)
        listRGB.append(rgbVals)

    # Determine the unit along the vertical slice since it's angled and
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

    if dumpResult:
        print('\nDumping values...')
        pickle.dump(tensors, open(case.resultPath + sliceName + '_rgbVals.p', 'wb'))
        pickle.dump(x2D, open(case.resultPath + sliceName + '_x2D.p', 'wb'))
        pickle.dump(y2D, open(case.resultPath + sliceName + '_y2D.p', 'wb'))
        pickle.dump(z2D, open(case.resultPath + sliceName + '_z2D.p', 'wb'))


    """
    Plotting
    """
    xLabel, yLabel = (r'$r$ [m]', r'$z$ [m]') \
        if case.slicesOrientate[sliceName] == 'vertical' else \
    (r'$x$ [m]', r'$y$ [m]')
    # Rotate the figure 90 deg clockwise, for 2D image plot
    rgbValsRot = ndimage.rotate(rgbVals, 90)
    figName = 'barycentric_' + sliceName
    if plotType in ('2D', '2d', 'all', 'All'):
        baryMapSlice = BaseFigure((None,), (None,), name = figName, xLabel = xLabel,
                                  yLabel = yLabel, save = save, show = show,
                                        figDir = case.resultPath)
        baryMapSlice.initializeFigure()
        # Real extent so that the axis have the correct AR
        extent = (np.min(r2D), np.max(r2D),
                  np.min(z2D),
                  np.max(z2D)) \
                  if case.slicesOrientate[sliceName] == 'vertical' else \
            (np.min(x2D), np.max(x2D),
                  np.min(y2D),
                  np.max(y2D))
        baryMapSlice.axes[0].imshow(rgbValsRot, origin = 'upper', aspect = 'equal', extent = extent)
        baryMapSlice.axes[0].set_xlabel(baryMapSlice.xLabel)
        baryMapSlice.axes[0].set_ylabel(baryMapSlice.yLabel)
        plt.tight_layout()
        plt.savefig(case.resultPath + figName + '.' + ext, dpi = dpi)

if plotType in ('3D', 'all'):
    baryMapSlice3D = PlotImageSlices3D(listX2D = listX2D, listY2D = listY2D, listZ2D = listZ2D, listRGB = listRGB,
                                       name = 'barycentric_' + str(
            sliceNames),
                                       xLabel = r'$x$ [m]',
                              yLabel = r'$y$ [m]', zLabel = r'$z$ [m]', save = save, show = show,
                                    figDir = case.resultPath, viewAngles = viewAngle, figWidth = figWidth, equalAxis =
                                       equalAxis)
    baryMapSlice3D.initializeFigure(figSize = figSize)
    baryMapSlice3D.plotFigure()
    baryMapSlice3D.finalizeFigure(tightLayout = True)

if save and plotType in ('2D', 'all'):
    print('\n2D plot of {0} saved at {1}'.format(sliceNames, case.resultPath))


"""
Plot Barycentric Color Map If Requested
"""
if showBaryExample:
    xTriLim, yTriLim = (0, 1), (0, np.sqrt(3) / 2.)
    verts = (
        (xTriLim[0], yTriLim[0]), (xTriLim[1], yTriLim[0]), (np.mean(xTriLim), yTriLim[1]),
        (xTriLim[0], yTriLim[0]))
    triangle = Path(verts)

    xTri, yTri = np.mgrid[xTriLim[0]:xTriLim[1]:1000j, yTriLim[0]:yTriLim[1]:1000j]
    xyTri = np.transpose((xTri.ravel(), yTri.ravel()))
    mask = triangle.contains_points(xyTri)
    # mask = mask.reshape(xTri.shape).T
    xyTri = xyTri[mask]
    # xTri, yTri = np.ma.array(xTri, mask, dtype = bool), np.ma.array(yTri, mask)

    c3 = xyTri[:, 1] / yTriLim[1]
    c1 = xyTri[:, 0] - 0.5 * c3
    c2 = 1 - c1 - c3

    rgbVals_example = np.vstack((c1, c2, c3)).T
    # rgbValsNew = np.empty((c1.shape[0], 3))
    # Each 2nd dim is an RGB array of the 2D grid
    rgbValsNew_example = (rgbVals_example + c_offset) ** c_exp

    baryMap3D = griddata(xyTri, rgbValsNew_example, (xTri, yTri))
    baryMap3D[np.isnan(baryMap3D)] = 1
    baryMap3D = ndimage.rotate(baryMap3D, 90)

    baryMapExample = BaseFigure((None,), (None,), name = baryMapExampleName, figDir = case.resultPath,
                                show = show, save = save)
    baryMapExample.initializeFigure()
    baryMapExample.axes[0].imshow(baryMap3D, origin = 'upper', aspect = 'equal', extent = (xTriLim[0], xTriLim[1],
                                                                                           yTriLim[0], yTriLim[1]))
    baryMapExample.axes[0].annotate(r'$\textbf{x}_{2c}$', (xTriLim[0], yTriLim[0]), (xTriLim[0] - 0.1, yTriLim[0]))
    baryMapExample.axes[0].annotate(r'$\textbf{x}_{3c}$', (np.mean(xTriLim), yTriLim[1]))
    baryMapExample.axes[0].annotate(r'$\textbf{x}_{1c}$', (xTriLim[1], yTriLim[0]))
    # baryMapExample.axes[0].get_yaxis().set_visible(False)
    # baryMapExample.axes[0].get_xaxis().set_visible(False)
    # baryMapExample.axes[0].set_axis_off()
    baryMapExample.axes[0].axis('off')
    plt.tight_layout()
    plt.savefig(case.resultPath + baryMapExampleName + '.' + ext, dpi = dpi)
    if save:
        print('\n{0} saved at {1}'.format(baryMapExampleName, case.resultPath))

    # baryPlot = PlotSurfaceSlices3D(x2D, y2D, z2D, (0,), show = True, name = 'bary', figDir = 'R:', save = True)
    # baryPlot.cmapLim = (0, 1)
    # baryPlot.cmapNorm = rgbVals
    # # baryPlot.cmapVals = plt.cm.ScalarMappable(norm = rgbVals, cmap = None)
    # baryPlot.cmapVals = rgbVals
    # # baryPlot.cmapVals.set_array([])
    # baryPlot.plot = baryPlot.cmapVals
    # baryPlot.initializeFigure()
    # baryPlot.axes[0].plot_surface(x2D, y2D, z2D, cstride = 1, rstride = 1, facecolors = rgbVals, vmin = 0, vmax = 1, shade = False)
    # baryPlot.finalizeFigure()
    #
    #
    #
    #
    #
    # print('\nDumping values...')
    # pickle.dump(tensors, open(case.resultPath + sliceName + '_tensors.p', 'wb'))
    # pickle.dump(case.slicesCoor[sliceName][:, 0], open(case.resultPath + sliceName + '_x.p', 'wb'))
    # pickle.dump(case.slicesCoor[sliceName][:, 1], open(case.resultPath + sliceName + '_y.p', 'wb'))
    # pickle.dump(case.slicesCoor[sliceName][:, 2], open(case.resultPath + sliceName + '_z.p', 'wb'))
    #
    # print('\nExecuting RGB_barycentric_colors_clean...')
    # import RBG_barycentric_colors_clean
    #
    #
    #
    #
    # # valsDecomp = case.mergeHorizontalComponents(valsDecomp)
