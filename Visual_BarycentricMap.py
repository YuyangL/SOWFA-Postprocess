import time as t
from PlottingTool import PlotSurfaceSlices3D
from PostProcess_SliceData import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy import ndimage
import pickle
import numpy as np

"""
User Inputs
"""
time = 'latest'
caseDir = 'J:'
caseDir = '/media/yluan/1'
caseName = 'ALM_N_H_ParTurb'
propertyName = 'uuPrime2'
sliceNames = 'alongWindRotorOne'
# Orientation of x-axis in x-y plane, in case of angled flow direction
# Only used for values decomposition
# Angle in rad and counter-clockwise
xOrientate = 6/np.pi

"""
Plot Settings
"""
precisionX, precisionY, precisionZ = 1000j, 1000j, 333j
interpMethod = 'nearest'
# Quiver plot is deprecated and refer to Visual_EigenVectors.py
plot = 'bary'  # 'bary', 'quiver'
showBaryExample = True
c_offset, c_exp = 0.65, 5.


"""
Read Slice Data
"""
case = SliceProperties(time = time, caseDir = caseDir, caseName = caseName, xOrientate = xOrientate)
case.readSlices(propertyName = propertyName, sliceNames = sliceNames)
# Go through specified slices
for sliceName in case.sliceNames:
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
    vals2D, tensors, eigValsGrid, eigVecsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor_Uninterpolated(vals2D, realizeIter = 0)
    t1 = t.time()
    print('\nFinished processAnisotropyTensor_Uninterpolated in {:.4f} s'.format(t1 - t0))

    # valsDecomp, tensors, eigValsGrid = case.processAnisotropyTensor_Fast(valsDecomp)
    # vals2D, tensors, eigValsGrid = case.processAnisotropyTensor_Uninterpolated(vals2D)


    """
    Interpolation
    """
    if plot == 'bary':
        xBary, yBary, rgbVals = case.getBarycentricMapCoordinates(eigValsGrid, c_offset = c_offset, c_exp = c_exp)

        x2D, y2D, z2D, rgbVals = \
            case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1], case.slicesCoor[sliceName][:, 2], rgbVals, sliceOrientate =
                                                     case.slicesOrientate[sliceName], xOrientate = case.xOrientate,
                                                     precisionX = precisionX, precisionY =
                                                     precisionY, precisionZ = precisionZ, interpMethod = interpMethod)
        print('\nDumping values...')
        pickle.dump(tensors, open(case.resultPath + sliceName + '_rgbVals.p', 'wb'))
        pickle.dump(x2D, open(case.resultPath + sliceName + '_x2D.p', 'wb'))
        pickle.dump(y2D, open(case.resultPath + sliceName + '_y2D.p', 'wb'))
        pickle.dump(z2D, open(case.resultPath + sliceName + '_z2D.p', 'wb'))

    elif plot == 'quiver':
        x2D, y2D, z2D, eigVecs3D = \
            case.interpolateDecomposedSliceData_Fast(case.slicesCoor[sliceName][:, 0], case.slicesCoor[sliceName][:, 1],
                                                     case.slicesCoor[sliceName][:, 2], eigVecsGrid[:, :, :, 0], sliceOrientate =
                                                     case.slicesOrientate[sliceName], xOrientate = case.xOrientate,
                                                     precisionX = precisionX, precisionY =
                                                     precisionY, precisionZ = precisionZ, interpMethod = interpMethod)


    """
    Plotting
    """
    if showBaryExample:
        xLim, yLim = (0, 1), (0, np.sqrt(3)/2.)
        verts = ((xLim[0], yLim[0]), (xLim[1], yLim[0]), (np.mean(xLim), yLim[1]), (xLim[0], yLim[0]))
        triangle = Path(verts)

        xTri, yTri = np.mgrid[xLim[0]:xLim[1]:100j, yLim[0]:yLim[1]:100j]
        xyTri = np.transpose((xTri.ravel(), yTri.ravel()))
        mask = triangle.contains_points(xyTri)
        # mask = mask.reshape(xTri.shape).T
        xyTri = xyTri[mask]
        # xTri, yTri = np.ma.array(xTri, mask, dtype = bool), np.ma.array(yTri, mask)

        c3 = xyTri[:, 1]/yLim[1]
        c1 = xyTri[:, 0] - 0.5*c3
        c2 = 1 - c1 - c3

        rgbVals = np.vstack((c1, c2, c3))
        # rgbValsNew = np.empty((c1.shape[0], 3))
        # Each 2nd dim is an RGB array of the 2D grid
        rgbValsNew = (rgbVals + c_offset)**c_exp

        baryMap3D = griddata(xyTri, rgbValsNew, (xTri, yTri))
        baryMap3D[np.isnan(baryMap3D)] = 1
        baryMap3D = ndimage.rotate(baryMap3D, 90)

        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(baryMap3D, origin = 'lower', aspect = 'equal')







    if plot == 'bary':
        # # Custom RGB colormap, with only red, green, and blue
        # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        # nBin = 1000
        # cmapName = 'rgb'
        # from matplotlib.colors import LinearSegmentedColormap
        # rgbCm = LinearSegmentedColormap.from_list(
        #         cmapName, colors, N = nBin)

        # rgbVals[rgbVals < 0] = 0.
        # rgbVals[rgbVals > 1] = 1.

        # Rotate the figure 90 deg clockwise
        rgbValsRot = ndimage.rotate(rgbVals, 90)
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(rgbValsRot, origin = 'upper', aspect = 'equal', extent = (0, 3000, 0, 1000))
        # fig.colorbar(im, ax = ax, extend = 'both')
        plt.savefig(case.resultPath + sliceName + '.png', dpi = 600)

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
    elif plot == 'quiver':
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.quiver(x2D, y2D, z2D, eigVecs3D[:, :, 0], eigVecs3D[:, :, 1], eigVecs3D[:, :, 2], length = 0.1, normalize = False)
