def plotLines2D(plot = 'line', txtFiles = (), X = None, Y = None, skipHeader = 0, xCol = 0, yCol = 1, xLabel = '$x$', yLabel = '$y$', lineLabels = 'Line',
           figDir = 'Results', xLim = None, yLim = None, invertY = False, noLegend = False, fontSize = 14,
           transparentBG = True, showGrid = False, show = True, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from Utilities import configurePlotSettings, convertDataTo2D
    from warnings import warn

    extraX_Keys = ('X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10')
    # A list of keys for extra lines to plot, max is 9 extra lines
    extraY_Keys = ('Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10')

    # If X and Y are both provided, then skip reading the data file
    if isinstance(X, (list, tuple, np.ndarray)) and isinstance(Y, (list, tuple, np.ndarray)):
        # Make sure X and Y are 2D array
        X = convertDataTo2D(X)
        Y = convertDataTo2D(Y)

        # If X and Y have the same size
        if X.shape == Y.shape:

            # If I provide a file/fig name
            if 'fileName' in kwargs:
                fileName = kwargs['fileName']
            else:
                fileName = 'UntitledData'

            # txtFiles now is a list of size 1 in order for the plot loop to work later
            txtFiles = [fileName]

            # Loop through extra Y if provided as kwarg
            for extraY_Key in extraY_Keys:
                if extraY_Key in kwargs:
                    # Append extra keys to txtFiles for the plot loop later
                    txtFiles.append(extraY_Key)
                    extraY = kwargs[extraY_Key]
                    extraY = convertDataTo2D(extraY)
                    # Concatenate along columns
                    Y = np.concatenate((Y, extraY), axis = 1)
                # Else, early stop
                else:
                    break

            for extraX_Key in extraX_Keys:
                if extraX_Key in kwargs:
                    extraX = kwargs[extraX_Key]
                    extraX = convertDataTo2D(extraX)
                    # Concatenate along columns
                    X = np.concatenate((X, extraX), axis = 1)
                # Else, early stop
                else:
                    Xtmp = X
                    for _ in range(Y.shape[1]):
                        Xtmp = np.concatenate((Xtmp, X), axis = 1)
                    X = Xtmp
                    break

            if 'useTex' in kwargs and isinstance(kwargs['useTex'], bool):
                lines, markers, colors = configurePlotSettings(lineCnt = len(txtFiles), useTex = kwargs['useTex'],
                                                           fontSize =
                fontSize)
            else:
                lines, markers, colors = configurePlotSettings(lineCnt = len(txtFiles), fontSize = fontSize)
        else:
            # Else early stop
            warn('\nError: [X] is provided but [X] and [Y] do not have the same size! Program aborted!\n', stacklevel
            = 2)
            return
    elif isinstance(txtFiles, (list, tuple)):
        lines, markers, colors = configurePlotSettings(len(txtFiles), fontSize = fontSize)
    elif isinstance(txtFiles, str):
        txtFiles = [txtFiles]
        lines, markers, colors = configurePlotSettings(fontSize = fontSize)
    else:
        warn('\nError: Invalid [txtFiles] or Invalid [X] and [Y]! Program aborted!\n', stacklevel = 2)
        return

    # Initialize figure
    fig = plt.figure()
    # Figure title
    fig.canvas.set_window_title(str(txtFiles))

    # Go through txtFiles list that was just created earlier and plot in this loop
    for idx, txtFile in enumerate(txtFiles):
        print('Plotting {0}...'.format(txtFile))
        # If X is None, then it means X and Y must have been in the txtFile provided
        if X is None:
            data = np.genfromtxt(txtFile + '.txt', skip_header = skipHeader)
            X = data[:, xCol]
            Y = data[:, yCol]

        # If the provided lineLabel is a list of labels and matches the number of lines to be plotted
        if isinstance(lineLabels, (list, tuple)) and len(lineLabels) == len(txtFiles):
            lineLabel = lineLabels[idx]
        # Else, lineLabel is probably just a prefix
        else:
            lineLabel = lineLabels + ' ' + str(idx)

        if plot == 'scatter':
            plt.scatter(X[:, idx], Y[:, idx], lw = 0, label = lineLabel, alpha = 0.5, marker = markers[idx],
                        color = colors[
                idx])
        else:
            if plot != 'line':
                warn('\nInvalid [plot] choice! Using default [line] plot...\n', stacklevel = 2)
            plt.plot(X[:, idx], Y[:, idx], ls = lines[idx], label = lineLabel, alpha = 0.75, color = colors[idx],
                     marker = 'o')

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if isinstance(xLim, (tuple, list)):
        plt.xlim(xLim)
    if isinstance(yLim, (tuple, list)):
        plt.ylim(yLim)

    # Reverse the y-axis for e.g. Cp if invertY is True
    if invertY:
        plt.gca().invert_yaxis()

    # If I specifically say noLegend is True, or if there's only one line, then don't show legend
    if noLegend or len(txtFiles) == 1:
        pass
    # Else, detect if multiple lines are provided, show legend
    elif len(txtFiles) > 1:
        if len(txtFiles) > 3:
            nCol = 2
        else:
            nCol = 1
        plt.legend(loc = 'best', shadow = True, fancybox = False, ncol = nCol)

    if showGrid:
        plt.grid(which = 'both')
    plt.tight_layout()
    plt.savefig(figDir + '/' + str(txtFiles) + '.pdf', transparent = transparentBG, format = 'pdf')

    if show:
        plt.show()


def plot2D(listX, listY, type = 'line', z2D = (None,), name = 'Untitled2D_Plot', fontSize = 14, plotLabels = (None,), alpha = 1., contourLvl = 10, cmap = 'plasma', xLabel = '$x$', yLabel = '$y$', zLabel = '$z$', figDir = './', show = True, xLim = (None,), yLim = (None,), xyScale = ('linear', 'linear'), saveFig = True, equalAxis = False):
    import matplotlib.pyplot as plt
    from Utilities import configurePlotSettings
    import numpy as np
    from warnings import warn

    if isinstance(listX, np.ndarray):
        listX, listY = (listX,), (listY,)

    # if len(listX) != len(listY):
    #     warn('\nThe number of provided x array does not match that of y. Not plotting!\n', stacklevel = 2)
    #     return

    if plotLabels[0] is None:
        plotLabels = (type,)*len(listX)

    lines, markers, colors = configurePlotSettings(lineCnt = len(listX), fontSize = fontSize)

    # plt.figure(name)
    fig, ax = plt.subplots(1, 1, num = name)
    if (z2D[0] is not None):
        x, y = np.array(listX[0]), np.array(listY[0])
        if len(np.array(x).shape) == 1:
            warn(
                '\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid automatically...\n',
                stacklevel = 2)
            x2D, y2D = np.meshgrid(x, y, sparse = True)
        else:
            x2D, y2D = x, y

        if type is 'contour':
            plt.contour(x2D, y2D, z2D, levels = contourLvl, cmap = cmap, extend = 'both')
        else:
            plt.contourf(x2D, y2D, z2D, levels = contourLvl, cmap = cmap, extend = 'both')
            
    else:
        for i in range(len(listX)):
            x, y = listX[i], listY[i]
            if type is 'scatter':
                plt.scatter(x, y, lw = 0, label = plotLabels[i], alpha = alpha, color = colors[i])
            else:
                plt.plot(x, y, ls = lines[i], label = plotLabels[i], color = colors[i], marker = markers[i], alpha = alpha)

    plt.xlabel(xLabel), plt.ylabel(yLabel)

    if equalAxis:
        ax.set_aspect('equal', 'box')

    if xLim[0] is not None:
        plt.xlim(xLim)

    if yLim[0] is not None:
        plt.ylim(yLim)

    plt.xscale(xyScale[0]), plt.yscale(xyScale[1])

    if len(listX) > 1:
        if len(listX) > 3:
            nCol = 2
        else:
            nCol = 1
        plt.legend(loc = 'best', shadow = True, fancybox = False, ncol = nCol)

    if z2D[0] is None:
        plt.grid(which = 'both', alpha = 0.5)
    else:
        cb = plt.colorbar(orientation = 'horizontal')
        cb.set_label(zLabel)
        cb.outline.set_visible(False)

    plt.box(False)

    # fig.canvas.draw()
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # ax.set_xticklabels(labels)
    # plt.annotate('ajsfbs', (1850, 0))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    plt.tight_layout()
    if saveFig:
        plt.savefig(figDir + '/' + name + '.png', transparent = True, bbox_inches = 'tight', dpi = 1000)

    if show:
        plt.show()


def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    # Draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    # loc1, loc2 : {1, 2, 3, 4}
    from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


def plot2DWithInsetZoom(listX, listY, zoomBox, type = 'line', z2D = (None,), name = 'Untitled2D_Plot', fontSize = 10, plotLabels = (None,),
           alpha = 1., contourLvl = 10, cmap = 'plasma', xLabel = '$x$', yLabel = '$y$', zLabel = '$z$', figDir = './',
           show = True, xLim = (None,), yLim = (None,), xyScale = ('linear', 'linear'), saveFig = True,
           equalAxis = False):
    import matplotlib.pyplot as plt
    from Utilities import configurePlotSettings
    import numpy as np
    from warnings import warn

    if isinstance(listX, np.ndarray):
        listX, listY = (listX,), (listY,)

    if len(listX) != len(listY):
        warn('\nThe number of provided x array does not match that of y. Not plotting!\n', stacklevel = 2)
        return

    if plotLabels[0] is None:
        plotLabels = (type,)*len(listX)

    lines, markers, colors = configurePlotSettings(lineCnt = len(listX), fontSize = fontSize)

    fig, ax = plt.subplots(nrows = 2, num = name, gridspec_kw = {'wspace':0, 'hspace':0})
    if (z2D[0] is not None):
        x, y = np.array(listX[0]), np.array(listY[0])
        if len(np.array(x).shape) == 1:
            warn(
                    '\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid '
                    'automatically...\n',
                    stacklevel = 2)
            x2D, y2D = np.meshgrid(x, y, sparse = True)
        else:
            x2D, y2D = x, y

        if type is 'contour':
            plot1 = ax[0].contour(x2D, y2D, z2D, levels = contourLvl, cmap = cmap, extend = 'both')
            plot2 = ax[1].contour(x2D, y2D, z2D, levels = contourLvl, cmap = cmap, extend = 'both')
        else:
            plot1 = ax[0].contourf(x2D, y2D, z2D, levels = contourLvl, cmap = cmap, extend = 'both')
            plot2 = ax[1].contourf(x2D, y2D, z2D, levels = contourLvl, cmap = cmap, extend = 'both')

    else:
        for i in range(len(listX)):
            x, y = listX[i], listY[i]
            if type is 'scatter':
                plot1 = ax[0].scatter(x, y, lw = 0, label = plotLabels[i], alpha = alpha, color = colors[i])
                plot2 = ax[1].scatter(x, y, lw = 0, label = plotLabels[i], alpha = alpha, color = colors[i])
            else:
                plot1 = ax[0].plot(x, y, ls = lines[i], label = plotLabels[i], color = colors[i], marker = markers[i],
                         alpha = alpha)
                plot2 = ax[1].scatter(x, y, lw = 0, label = plotLabels[i], alpha = alpha, color = colors[i])

    ax[1].set_xlim((zoomBox[0], zoomBox[1]))
    ax[1].set_ylim((zoomBox[2], zoomBox[3]))

    ax[1].set_xlabel(xLabel), ax[0].set_ylabel(yLabel), ax[1].set_ylabel(yLabel)

    if equalAxis:
        ax[0].set_aspect('equal', 'box')
        ax[1].set_aspect('equal', 'box')

    if xLim[0] is not None:
        ax[0].set_xlim(xLim)

    if yLim[0] is not None:
        ax[0].set_ylim(yLim)

    plt.xscale(xyScale[0]), plt.yscale(xyScale[1])

    if len(listX) > 1:
        if len(listX) > 3:
            nCol = 2
        else:
            nCol = 1
        plt.legend(loc = 'best', shadow = True, fancybox = False, ncol = nCol)

    mark_inset(ax[0], ax[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none", ec = "gray", ls = ':')

    # fig.canvas.draw()
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # ax.set_xticklabels(labels)
    # plt.annotate('ajsfbs', (1850, 0))

    ax[0].spines['top'].set_visible(False), ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False), ax[0].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_color('gray'), ax[1].spines['top'].set_color('gray')
    ax[1].spines['right'].set_color('gray'), ax[1].spines['left'].set_color('gray')
    ax[1].spines['bottom'].set_linestyle(':'), ax[1].spines['top'].set_linestyle(':')
    ax[1].spines['right'].set_linestyle(':'), ax[1].spines['left'].set_linestyle(':')

    plt.draw()

    plt.tight_layout()

    if z2D[0] is None:
        plt.grid(which = 'both', alpha = 0.5)
    else:
        fig.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cb = plt.colorbar(plot1, cax = cbar_ax, orientation = 'vertical')
        # cb = plt.colorbar(plot1, orientation = 'vertical')
        cb.set_label(zLabel)
        cb.outline.set_visible(False)

    if saveFig:
        plt.savefig(figDir + '/' + name + '.png', transparent = True, bbox_inches = 'tight', dpi = 1000)

    if show:
        plt.show()


def plotSlices3D(listSlices2D, x2D, y2D, sliceOffsets, zDir = 'z', name = 'UntitledSlices3D', figDir = './', alpha = 1, nContour = 20,
                 fontSize = 14, viewAngles = (35, -120), show = True, saveFig = True):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from warnings import warn
    import numpy as np
    from Utilities import configurePlotSettings

    _, _, _ = configurePlotSettings(fontSize = fontSize)

    sliceOffsetsIter = iter(sliceOffsets)
    # # Numpy array treatment
    # for i in range(len(listSlices2D)):
    #     listSlices2D[i] = np.array(listSlices2D[i])


    ax = plt.figure(name).gca(projection = '3d')
    # Multiplier to stretch the 'z' axis
    # When 1 slice, no stretch
    # When 3 slices, 1.3 in z, and 0.65 in other directions
    figZ = 0.15*len(listSlices2D) + 0.85
    figXY = -0.175*len(listSlices2D) + 1.175
    if zDir is 'y':
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([figXY, figZ, figXY, 1]))
    elif zDir is 'x':
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([figZ, figXY, figXY, 1]))
    else:
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([figXY, figXY, figZ, 1]))


    for slice in listSlices2D:
        plot = ax.contourf(x2D, y2D, slice, nContour, zdir = zDir, offset = next(sliceOffsetsIter), alpha = alpha)

    ax.set_zlim(min(sliceOffsets), max(sliceOffsets))
    ax.set_xticks(np.arange(int(x2D.min()), int(x2D.max()) + 1, int(x2D.max())/3))
    ax.set_yticks(np.arange(int(y2D.min()), int(y2D.max()) + 1, int(y2D.max())/3))
    ax.view_init(viewAngles[0], viewAngles[1])
    cbaxes = ax.add_axes([0.8, 0.1, 0.02, 0.8])
    cb = plt.colorbar(plot, cax = cbaxes, drawedges = False, extend = 'both')
    # Remove colorbar outline
    cb.outline.set_linewidth(0)
    # Turn off background on all three panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.tight_layout()
    if saveFig:
        plt.savefig(figDir + '/' + name + '.png', dpi = 1000, transparent = True, bbox_inches = 'tight')

    if show:
        plt.show()


def setAxesEqual3D(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    import numpy as np

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plotIsosurfaces3D(x3D, y3D, z3D, surface3Dlist, contourList, slice3Dlist = (None,), boundSurface = (None,),
customColors = (None,), figDir = './', name = 'UntitledIsosurface', sliceOriantations = (None,), sliceOffsets = (None,), boundSlice = (None,), sliceValRange = (None,)):
    from mayavi import mlab
    from mayavi.modules.contour_grid_plane import ContourGridPlane
    from mayavi.modules.outline import Outline
    from mayavi.api import Engine
    import numpy as np
    from warnings import warn

    x3D, y3D, z3D = np.array(x3D), np.array(y3D), np.array(z3D)

    engine = Engine()
    engine.start()
    # if len(engine.scenes) == 0:
    #     engine.new_scene()

    if customColors[0] is not None:
        customColors = iter(customColors)
    else:
        customColors = iter(('inferno', 'viridis', 'coolwarm', 'Reds', 'Blues')*2)

    contourList = iter(contourList)
    if sliceOriantations[0] is not None:
        sliceOriantations = iter(sliceOriantations)
    else:
        sliceOriantations = iter(('z_axes',)*len(slice3Dlist))

    if boundSurface[0] is None:
        boundSurface = (x3D.min(), x3D.max(), y3D.min(), y3D.max(), z3D.min(), z3D.max())

    if boundSlice[0] is None:
        boundSlice = (x3D.min(), x3D.max(), y3D.min(), y3D.max(), z3D.min(), z3D.max())

    if sliceOffsets[0] is None:
        sliceOffsets = iter((0,)*len(slice3Dlist))
    else:
        sliceOffsets = iter(sliceOffsets)

    if sliceValRange[0] is None:
        sliceValRange = iter((None, None)*len(slice3Dlist))
    else:
        sliceValRange = iter(sliceValRange)


    mlab.figure(name, engine = engine, size = (1200, 900), bgcolor = (1, 1, 1), fgcolor = (0.5, 0.5, 0.5))

    for surface3D in surface3Dlist:
        color = next(customColors)
        # If a colormap given
        if isinstance(color, str):
            mlab.contour3d(x3D, y3D, z3D, surface3D, contours = next(contourList), colormap = color,
                           extent = boundSurface)
        # If only one color of (0-1, 0-1, 0-1) given
        else:
            mlab.contour3d(x3D, y3D, z3D, surface3D, contours = next(contourList), color = color,
                           extent = boundSurface)

        # cgp = ContourGridPlane()
        # engine.add_module(cgp)
        # cgp.grid_plane.axis = 'y'
        # cgp.grid_plane.position = x3D.shape[1] - 1
        # cgp.contour.number_of_contours = 20
        # # contour_grid_plane2.actor.mapper.scalar_range = array([298., 302.])
        # # contour_grid_plane2.actor.mapper.progress = 1.0
        # # contour_grid_plane2.actor.mapper.scalar_mode = 'use_cell_data'
        # cgp.contour.filled_contours = True
        # cgp.actor.property.lighting = False

    for slice3D in slice3Dlist:
        sliceOriantation = next(sliceOriantations)
        sliceOffset = next(sliceOffsets)
        if sliceOriantation == 'z_axes':
            origin = np.array([boundSlice[0], boundSlice[2], sliceOffset])
            point1 = np.array([boundSlice[1], boundSlice[2], sliceOffset])
            point2 = np.array([boundSlice[0], boundSlice[3], sliceOffset])
        elif sliceOriantation == 'x_axes':
            origin = np.array([sliceOffset, boundSlice[2], boundSlice[4]])
            point1 = np.array([sliceOffset, boundSlice[3], boundSlice[4]])
            point2 = np.array([sliceOffset, boundSlice[2], boundSlice[5]])
        else:
            origin = np.array([boundSlice[0], sliceOffset, boundSlice[4]])
            point1 = np.array([boundSlice[1], sliceOffset, boundSlice[4]])
            point2 = np.array([boundSlice[0], sliceOffset, boundSlice[5]])

        image_plane_widget = mlab.volume_slice(x3D, y3D, z3D, slice3D, plane_orientation = sliceOriantation, colormap = next(customColors), vmin = next(sliceValRange), vmax = next(sliceValRange))
        image_plane_widget.ipw.reslice_interpolate = 'cubic'
        image_plane_widget.ipw.origin = origin
        image_plane_widget.ipw.point1 = point1
        image_plane_widget.ipw.point2 = point2
        # image_plane_widget.ipw.slice_index = 2
        image_plane_widget.ipw.slice_position = sliceOffset

    # Contour grid plane at last y if the last slice is in xy plane
    if sliceOriantation == 'z_axes':
        cgp2 = ContourGridPlane()
        engine.add_module(cgp2)
        cgp2.grid_plane.axis = 'y'
        cgp2.grid_plane.position = x3D.shape[1] - 1
        cgp2.contour.number_of_contours = 20
        cgp2.contour.filled_contours = True
        cgp2.actor.property.lighting = False

    outline = Outline()
    engine.add_module(outline)
    outline.actor.property.color = (0.2, 0.2, 0.2)
    outline.bounds = np.array(boundSurface)
    outline.manual_bounds = True

    mlab.view(azimuth = 270, elevation = 45)
    mlab.move(-500, 0, 0)
    mlab.savefig(figDir + '/' + name + '.png', magnification = 3)
    print('\nFigure ' + name + ' saved at ' + figDir)
    mlab.show()









