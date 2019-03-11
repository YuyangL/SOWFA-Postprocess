import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from warnings import warn

class BaseFigure:
    def __init__(self, listX, listY, name = 'UntitledFigure', fontSize = 8, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, save = True, equalAxis = False, xLim = (None,), yLim = (None,), figWidth = 'half', figHeightMultiplier = 1., subplots = (1, 1), colors = ('tableau10',)):
        (self.listX, self.listY) = ((listX,), (listY,)) if isinstance(listX, np.ndarray) else (listX, listY)
        self.name, self.figDir, self.save, self.show = name, figDir, save, show
        if not self.show:
            plt.ioff()
        else:
            plt.ion()

        self.xLabel, self.yLabel, self.equalAxis = xLabel, yLabel, equalAxis
        self.xLim, self.yLim = xLim, yLim
        (self.colors, self.gray) = self.setColors(which = colors[0]) if colors[0] in ('tableau10', 'tableau20') else (colors, (89/255., 89/255., 89/255.))

        self.subplots, self.figWidth, self.figHeightMultiplier, self.fontSize = subplots, figWidth, figHeightMultiplier, fontSize


    @staticmethod
    def setColors(which = 'qualitative'):
        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (23, 190, 207), (214, 39, 40), (188, 189, 34), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)]
        # Orange, blue, magenta, cyan, red, teal, grey
        qualitative = [(238, 119, 51), (0, 119, 187), (238, 51, 119), (51, 187, 238), (204, 51, 117), (0, 153, 136), (187, 187, 187)]
        colorsDict = {'tableau20': tableau20,
                      'tableau10': tableau10,
                      'qualitative': qualitative}
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        colors = colorsDict[which]
        for i in range(len(colors)):
            r, g, b = colors[i]
            colors[i] = (r/255., g/255., b/255.)

        tableauGray = (89/255., 89/255., 89/255.)
        return colors, tableauGray


    @staticmethod
    def latexify(fig_width = None, fig_height = None, figWidth = 'half', linewidth = 1, fontSize = 8, subplots = (1, 1), figHeightMultiplier = 1.):
        """Set up matplotlib's RC params for LaTeX plotting.
        Call this before plotting a figure.

        Parameters
        ----------
        fig_width : float, optional, inches
        fig_height : float,  optional, inches
        """
        # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
        if fig_width is None:
            if subplots[1] == 1:
                fig_width = 3.39 if figWidth is 'half' else 6.9  # inches
            else:
                fig_width = 6.9  # inches

        if fig_height is None:
            golden_mean = (np.sqrt(5) - 1.0)/2.0  # Aesthetic ratio
            # In case subplots option is not applicable e.g. normal Plot2D and you still want elongated height
            fig_height = fig_width*golden_mean*figHeightMultiplier  # height in inches
            fig_height *= (0.25 + (subplots[0] - 1)) if subplots[0] > 1 else 1

        MAX_HEIGHT_INCHES = 8.0
        if fig_height > MAX_HEIGHT_INCHES:
            warn("\nfig_height too large:" + str(fig_height) +
                  ". Will reduce to " + str(MAX_HEIGHT_INCHES) + " inches", stacklevel = 2)
            fig_height = MAX_HEIGHT_INCHES

        tableauGray = (89/255., 89/255., 89/255.)
        mpl.rcParams.update({
            'backend':             'ps',
            'text.latex.preamble': [r"\usepackage{gensymb,amsmath}"],
            'axes.labelsize':      fontSize,  # fontsize for x and y labels (was 10)
            'axes.titlesize':      fontSize + 2.,
            'font.size':           fontSize,  # was 10
            'legend.fontsize':     fontSize - 2.,  # was 10
            'xtick.labelsize':     fontSize - 2.,
            'ytick.labelsize':     fontSize - 2.,
            'xtick.color':         tableauGray,
            'ytick.color':         tableauGray,
            'xtick.direction':     'out',
            'ytick.direction':     'out',
            'text.usetex':         True,
            'figure.figsize':      (fig_width, fig_height),
            'font.family':         'serif',
            "legend.framealpha":   0.5,
            'legend.edgecolor':    'none',
            'lines.linewidth':     linewidth,
            'lines.markersize':    2,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            'axes.edgecolor':      tableauGray,
            'lines.antialiased':   True,
            'patch.antialiased':   True,
            'text.antialiased':    True})


    def initializeFigure(self):
        self.latexify(fontSize = self.fontSize, figWidth = self.figWidth, subplots = self.subplots, figHeightMultiplier = self.figHeightMultiplier)

        self.fig, self.axes = plt.subplots(self.subplots[0], self.subplots[1], num = self.name)
        self.axes = (self.axes,) if not isinstance(self.axes, np.ndarray) else self.axes
        print('\nFigure ' + self.name + ' initialized')


    def plotFigure(self):
        print('\nPlotting ' + self.name + '...')


    def _ensureMeshGrid(self):
        if len(np.array(self.listX[0]).shape) == 1:
            warn('\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid '
                    'automatically...\n',
                    stacklevel = 2)
            # Convert tuple to list
            self.listX, self.listY = list(self.listX), list(self.listY)
            self.listX[0], self.listY[0] = np.meshgrid(self.listX[0], self.listY[0], sparse = False)


    def finalizeFigure(self, xyScale = (None,), tightLayout = True, setXYlabel = (True, True), grid = True, transparentBg = False, legLoc = 'best'):
        if len(self.listX) > 1:
            nCol = 2 if len(self.listX) > 3 else 1
            self.axes[0].legend(loc = legLoc, shadow = False, fancybox = False, ncol = nCol)

        if grid:
            self.axes[0].grid(which = 'major', alpha = 0.25)

        if setXYlabel[0]:
            self.axes[0].set_xlabel(self.xLabel)

        if setXYlabel[1]:
            self.axes[0].set_ylabel(self.yLabel)

        if self.equalAxis:
            # Only execute 2D equal axis if the figure is acutally 2D
            try:
                self.viewAngles
            except AttributeError:
                self.axes[0].set_aspect('equal', 'box')

        if self.xLim[0] is not None:
            self.axes[0].set_xlim(self.xLim)

        if self.yLim[0] is not None:
            self.axes[0].set_ylim(self.yLim)

        if xyScale[0] is not None:
            self.axes[0].set_xscale(xyScale[0]), self.axes[0].set_yscale(xyScale[1])

        if tightLayout:
            plt.tight_layout()

        print('\nFigure ' + self.name + ' finalized')
        if self.save:
            # plt.savefig(self.figDir + '/' + self.name + '.png', transparent = transparentBg, bbox_inches = 'tight', dpi = 1000)
            plt.savefig(self.figDir + '/' + self.name + '.png', transparent = transparentBg,
                        dpi = 1000)
            print('\nFigure ' + self.name + '.png saved in ' + self.figDir)

        if self.show:
            plt.show()
        # Close current figure window
        # so that the next figure will be based on a new figure window even if the same name
        else:
            plt.close()


class Plot2D(BaseFigure):
    def __init__(self, listX, listY, z2D = (None,), type = 'infer', alpha = 0.75, zLabel = '$z$', cmap = 'plasma', gradientBg = False, gradientBgRange = (None, None), gradientBgDir = 'x', **kwargs):
        self.z2D = z2D
        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'D', 'v', '^', '<', '>', 's', '8', 'p')*3
        self.alpha, self.cmap = alpha, cmap
        self.zLabel = zLabel
        self.gradientBg, self.gradientBgRange, self.gradientBgDir = gradientBg, gradientBgRange, gradientBgDir

        super().__init__(listX, listY, **kwargs)

        # If multiple data provided, make sure type is a tuple of the same length
        if type == 'infer':
            self.type = ('contourf',)*len(listX) if z2D[0] is not None else ('line',)*len(listX)
        else:
            self.type = (type,)*len(listX) if isinstance(type, str) else type


    def plotFigure(self, plotsLabel = (None,), contourLvl = 10):
        # Gradient background, only for line and scatter plots
        if self.gradientBg and self.type[0] in ('line', 'scatter'):
            x2D, y2D = np.meshgrid(np.linspace(self.xLim[0], self.xLim[1], 3), np.linspace(self.yLim[0], self.yLim[1], 3))
            z2D = (np.meshgrid(np.linspace(self.xLim[0], self.xLim[1], 3), np.arange(3)))[0] if self.gradientBgDir is 'x' else (np.meshgrid(np.arange(3), np.linspace(self.yLim[0], self.yLim[1], 3)))[1]
            self.axes[0].contourf(x2D, y2D, z2D, 500, cmap = 'gray', alpha = 0.33, vmin = self.gradientBgRange[0], vmax = self.gradientBgRange[1])

        super().plotFigure()

        self.plotsLabel = np.arange(1, len(self.listX) + 1) if plotsLabel[0] is None else plotsLabel
        self.plots = [None]*len(self.listX)
        for i in range(len(self.listX)):
            if self.type[i] == 'line':
                self.plots[i] = self.axes[0].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = str(self.plotsLabel[i]), color = self.colors[i], alpha = self.alpha)
            elif self.type[i] == 'scatter':
                self.plots[i] = self.axes[0].scatter(self.listX[i], self.listY[i], lw = 0, label = str(self.plotsLabel[i]), alpha = self.alpha, color = self.colors[i], marker = self.markers[i])
            elif self.type[i] == 'contourf':
                self._ensureMeshGrid()
                self.plots[i] = self.axes[0].contourf(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both', antialiased = False)
            elif self.type[i] == 'contour':
                self._ensureMeshGrid()
                self.plots[i] = self.axes[0].contour(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            else:
                warn("\nUnrecognized plot type! type must be one/list of ('infer', 'line', 'scatter', 'contourf', 'contour').\n", stacklevel = 2)
                return


    def finalizeFigure(self, cbarOrientate = 'horizontal', **kwargs):
        if self.type in ('contourf', 'contour') and len(self.axes) == 1:
            cb = plt.colorbar(self.plots[0], ax = self.axes[0], orientation = cbarOrientate)
            cb.set_label(self.zLabel)
            super().finalizeFigure(grid = False, **kwargs)
        else:
            super().finalizeFigure(**kwargs)



class Plot2D_InsetZoom(Plot2D):
    def __init__(self, listX, listY, zoomBox, subplots = (2, 1), **kwargs):
        super().__init__(listX, listY, figWidth = 'full', subplots = subplots, **kwargs)

        self.zoomBox = zoomBox


    @staticmethod
    def _mark_inset(parent_axes, inset_axes, loc1a = 1, loc1b = 1, loc2a = 2, loc2b = 2, **kwargs):
        # Draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        # loc1, loc2 : {1, 2, 3, 4}
        rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

        pp = BboxPatch(rect, fill = False, **kwargs)
        parent_axes.add_patch(pp)

        p1 = BboxConnector(inset_axes.bbox, rect, loc1 = loc1a, loc2 = loc1b, **kwargs)
        inset_axes.add_patch(p1)
        p1.set_clip_on(False)
        p2 = BboxConnector(inset_axes.bbox, rect, loc1 = loc2a, loc2 = loc2b, **kwargs)
        inset_axes.add_patch(p2)
        p2.set_clip_on(False)

        print('\nInset created')
        return pp, p1, p2


    def plotFigure(self, plotsLabel = (None,), contourLvl = 10):
        super().plotFigure(plotsLabel, contourLvl)
        for i in range(len(self.listX)):
            if self.type is 'line':
                self.axes[1].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = str(self.plotsLabel[i]), alpha = self.alpha, color = self.colors[i])
            elif self.type is 'scatter':
                self.axes[1].scatter(self.listX[i], self.listY[i], lw = 0, label = str(self.plotsLabel[i]), alpha = self.alpha, marker = self.markers[i])
            elif self.type is 'contourf':
                self.axes[1].contourf(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            elif self.type is 'contour':
                self.axes[1].contour(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')


    def finalizeFigure(self, cbarOrientate = 'vertical', setXYlabel = (False, True), xyScale = ('linear', 'linear'), **kwargs):
        self.axes[1].set_xlim(self.zoomBox[0], self.zoomBox[1]), self.axes[1].set_ylim(self.zoomBox[2], self.zoomBox[3])
        self.axes[1].set_xlabel(self.xLabel), self.axes[1].set_ylabel(self.yLabel)
        if self.equalAxis:
            self.axes[1].set_aspect('equal', 'box')

        self.axes[1].set_xscale(xyScale[0]), self.axes[1].set_yscale(xyScale[1])
        self._mark_inset(self.axes[0], self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none",
                         ec = self.gray, ls = ':')
        if self.type in ('contour', 'contourf'):
            for ax in self.axes:
                ax.tick_params(axis = 'both', direction = 'out')

        else:
            self.axes[1].grid(which = 'both', alpha = 0.25)
            if len(self.listX) > 1:
                nCol = 2 if len(self.listX) > 3 else 1
                self.axes[1].legend(loc = 'best', shadow = False, fancybox = False, ncol = nCol)

        for spine in ('top', 'bottom', 'left', 'right'):
            if self.type in ('contour', 'contourf'):
                self.axes[0].spines[spine].set_visible(False)
            self.axes[1].spines[spine].set_visible(True)
            self.axes[1].spines[spine].set_linestyle(':')

        # plt.draw()
        # Single colorbar
        if self.type in ('contour', 'contourf'):
            self.fig.subplots_adjust(bottom = 0.1, top = 0.9, left = 0.1, right = 0.8)  # , wspace = 0.02, hspace = 0.2)
            cbar_ax = self.fig.add_axes((0.83, 0.1, 0.02, 0.8))
            cb = plt.colorbar(self.plots[0], cax = cbar_ax, orientation = 'vertical')
            cb.set_label(self.zLabel)
            cb.ax.tick_params(axis = 'y', direction = 'out')

        super().finalizeFigure(tightLayout = False, cbarOrientate = cbarOrientate, setXYlabel = setXYlabel, xyScale = xyScale, grid = False, **kwargs)


class BaseFigure3D(BaseFigure):
    def __init__(self, listX2D, listY2D, zLabel = '$z$', alpha = 1, viewAngles = (15, -115), zLim = (None,), cmap = 'plasma', cmapLabel = '$U$', grid = True, cbarOrientate = 'horizontal', **kwargs):
        super(BaseFigure3D, self).__init__(listX = listX2D, listY = listY2D, **kwargs)

        self.zLabel, self.zLim = zLabel, zLim
        self.cmapLabel, self.cmap = cmapLabel, cmap
        self.alpha, self.grid, self.viewAngles = alpha, grid, viewAngles
        self.plot, self.cbarOrientate = None, cbarOrientate


    def initializeFigure(self, figSize = (1, 1)):
        self.latexify(fontSize = self.fontSize, figWidth = self.figWidth, subplots = figSize)

        self.fig = plt.figure(self.name)
        self.axes = (self.fig.gca(projection = '3d'),)


    def plotFigure(self):
        super(BaseFigure3D, self).plotFigure()

        self._ensureMeshGrid()


    def finalizeFigure(self, fraction = 0.06, pad = 0.08, showCbar = True, reduceNtick = True, **kwargs):
        self.axes[0].set_zlabel(self.zLabel)
        if self.zLim[0] is not None:
            self.axes[0].set_zlim(self.zLim)

        if showCbar:
            # cbaxes = self.fig.add_axes([0.1, 0.1, 0.03, 0.8])
            # cb = plt.colorbar(self.plot, cax = cbaxes)
            cb = plt.colorbar(self.plot, fraction = fraction, pad = pad, orientation = self.cbarOrientate, extend = 'both', aspect = 25, shrink = 0.75)
            cb.set_label(self.cmapLabel)

        # Turn off background on all three panes
        self.format3D_Axes(self.axes[0])

        # [REQUIRES SOURCE CODE MODIFICATION] Equal axis
        # Edit the get_proj function inside site-packages\mpl_toolkits\mplot3d\axes3d.py:
        # try: self.localPbAspect=self.pbaspect
        # except AttributeError: self.localPbAspect=[1,1,1]
        # xmin, xmax = np.divide(self.get_xlim3d(), self.pbaspect[0])
        # ymin, ymax = np.divide(self.get_ylim3d(), self.pbaspect[1])
        # zmin, zmax = np.divide(self.get_zlim3d(), self.pbaspect[2])
        xLim = self.axes[0].get_xlim() if self.xLim[0] is None else self.xLim
        yLim = self.axes[0].get_ylim() if self.yLim[0] is None else self.yLim
        zLim = self.axes[0].get_zlim() if self.zLim[0] is None else self.zLim
        if self.equalAxis:
            try:
                arZX = abs((zLim[1] - zLim[0])/(xLim[1] - xLim[0]))
                arYX = abs((yLim[1] - yLim[0])/(xLim[1] - xLim[0]))
                # Axes aspect ratio doesn't really work properly
                pbaspect = (1., arYX, arZX*2)
                self.axes[0].pbaspect = pbaspect  # (1, ar, 1) if zDir is 'z' else (1, 1, ar)
            except AttributeError:
                warn('\nTo set custom aspect ratio of the 3D plot, you need modification of the source code axes3d.py. The aspect ratio might be incorrect for ' + self.name + '\n', stacklevel = 2)
                pass

        if reduceNtick:
            self.axes[0].set_xticks(np.linspace(xLim[0], xLim[1], 3))
            self.axes[0].set_yticks(np.linspace(yLim[0], yLim[1], 3))
            self.axes[0].set_zticks(np.linspace(zLim[0], zLim[1], 3))

        # # Strictly equal axis of all three axis
        # _, _, _, _, _, _ = self.get3D_AxesLimits(self.axes[0])
        # 3D grid
        self.axes[0].grid(self.grid)
        self.axes[0].view_init(self.viewAngles[0], self.viewAngles[1])
        # # View distance
        # self.axes[0].dist = 11

        super().finalizeFigure(grid = False, tightLayout = False, **kwargs)


    @staticmethod
    def format3D_Axes(ax):
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_yaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_zaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})


    @staticmethod
    def get3D_AxesLimits(ax, setAxesEqual = True):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a Matplotlib axis, e.g., as output from plt.gca().
        '''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        # The plot bounding box is a sphere in the sense of the infinity norm,
        # hence I call half the max range the plot radius.
        if setAxesEqual:
            plot_radius = 0.5*max([x_range, y_range, z_range])
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([0, z_middle + plot_radius])

        return x_range, y_range, z_range, x_limits, y_limits, z_limits


class PlotContourSlices3D(BaseFigure3D):
    def __init__(self, contourX2D, contourY2D, listSlices2D, sliceOffsets, zDir = 'z', contourLvl = 10, gradientBg = True, **kwargs):
        super(PlotContourSlices3D, self).__init__(listX2D = (contourX2D,), listY2D = (contourY2D,), **kwargs)

        self.listSlices2D = iter((listSlices2D,)) if isinstance(listSlices2D, np.ndarray) else iter(listSlices2D)
        self.sliceOffsets, self.zDir = iter(sliceOffsets), zDir
        self.xLim = (min(sliceOffsets), max(sliceOffsets)) if (self.xLim[0] is None) and (zDir is 'x') else self.xLim
        self.yLim = (min(sliceOffsets), max(sliceOffsets)) if (self.yLim[0] is None) and (zDir is 'y') else self.yLim
        self.zLim = (min(sliceOffsets), max(sliceOffsets)) if (self.zLim[0] is None) and (zDir is 'z') else self.zLim
        # If axis limits are still not set, infer
        if self.xLim[0] is None:
            self.xLim = (np.min(contourX2D), np.max(contourX2D))

        if self.yLim[0] is None:
            self.yLim = (np.min(contourY2D), np.max(contourY2D))

        if self.zLim[0] is None:
            self.zLim = (np.min(listSlices2D), np.max(listSlices2D))

        self.sliceMin, self.sliceMax = np.min(listSlices2D), np.max(listSlices2D)
        self.contourLvl, self.gradientBg = contourLvl, gradientBg
        # Good initial view angle
        self.viewAngles = (20, -115) if zDir is 'z' else (15, -60)
        self.cbarOrientate = 'vertical' if zDir is 'z' else 'horizontal'


    def initializeFigure(self, **kwargs):
        # If zDir is 'z', then the figure height is twice width, else, figure width is twice height
        figSize = (2.75, 1) if self.zDir is 'z' else (1, 2)

        super().initializeFigure(figSize = figSize)


    def plotFigure(self):
        super(PlotContourSlices3D, self).plotFigure()

        # Currently, gradient background feature is only available for zDir = 'x'
        if self.gradientBg:
            if self.zDir is 'x':
                x2Dbg, y2Dbg = np.meshgrid(np.linspace(self.xLim[0], self.xLim[1], 3), np.linspace(self.yLim[0], self.yLim[1], 3))
                z2Dbg, _ = np.meshgrid(np.linspace(self.xLim[0], self.xLim[1], 3), np.linspace(self.zLim[0], self.zLim[1], 3))
                self.axes[0].contourf(x2Dbg, y2Dbg, z2Dbg, 500, zdir = 'z', offset = 0, cmap = 'gray', alpha = 0.5, antialiased = True)
                # # Uncomment below to enable gradient background of all three planes
                # self.axes[0].contourf(x2Dbg, z2Dbg, y2Dbg, 500, zdir = 'y', offset = 300, cmap = 'gray', alpha = 0.5, antialiased = True)
                # Y3, Z3 = np.meshgrid(np.linspace(self.yLim[0], self.yLim[1], 3),
                #                      np.linspace(self.zLim[0], self.zLim[1], 3))
                # X3 = np.ones(Y3.shape)*self.xLim[0]
                # self.axes[0].plot_surface(X3, Y3, Z3, color = 'gray', alpha = 0.5)
            else:
                warn('\nGradient background only supports zDir = "x"!\n', stacklevel = 2)

        # Actual slice plots
        for slice in self.listSlices2D:
            if self.zDir is 'x':
                X, Y, Z = slice, self.listX[0], self.listY[0]
            elif self.zDir is 'y':
                X, Y, Z = self.listX[0], slice, self.listY[0]
            else:
                X, Y, Z = self.listX[0], self.listY[0], slice

            # "levels" makes sure all slices are in same cmap range
            self.plot = self.axes[0].contourf(X, Y, Z, self.contourLvl, zdir = self.zDir,
                                              offset = next(self.sliceOffsets), alpha = self.alpha, cmap = self.cmap,
                                              levels = np.linspace(self.sliceMin, self.sliceMax, 100), antialiased = False)


    def finalizeFigure(self, **kwargs):
        # Custom color bar location in the figure
        (fraction, pad) = (0.046, 0.04) if self.zDir is 'z' else (0.06, 0.08)
        # if self.zDir is 'z':
        #     ar = abs((self.yLim[1] - self.yLim[0])/(self.xLim[1] - self.xLim[0]))
        #     pbaspect = (1, ar, 1)
        # elif self.zDir is 'x':
        #     ar = abs((self.zLim[1] - self.zLim[0])/(self.yLim[1] - self.yLim[0]))
        #     pbaspect = (1, 1, ar)
        # else:
        #     ar = abs((self.zLim[1] - self.zLim[0])/(self.xLim[1] - self.xLim[0]))
        #     pbaspect = (1, 1, ar)

        super(PlotContourSlices3D, self).finalizeFigure(fraction = fraction, pad = pad, **kwargs)


class PlotSurfaceSlices3D(BaseFigure3D):
    def __init__(self, listX2D, listY2D, listZ2D, listSlices2D, **kwargs):
        super(PlotSurfaceSlices3D, self).__init__(listX2D = listX2D, listY2D = listY2D, **kwargs)

        self.xLim = (np.min(listX2D), np.max(listX2D)) if self.xLim[0] is None else self.xLim
        self.yLim = (np.min(listY2D), np.max(listY2D)) if self.yLim[0] is None else self.yLim
        self.zLim = (np.min(listZ2D), np.max(listZ2D)) if self.zLim[0] is None else self.zLim
        self.listX2D, self.listY2D = iter(self.listX), iter(self.listY)
        self.listZ2D = iter((listZ2D,)) if isinstance(listZ2D, np.ndarray) else iter(listZ2D)
        # Find minimum and maximum of the slices values for color, ignore NaN
        self.cmapLim = (np.nanmin(listSlices2D), np.nanmax(listSlices2D))
        self.listSlices2D = iter((listSlices2D,)) if isinstance(listSlices2D, np.ndarray) else iter(listSlices2D)

        self.cmapNorm = mpl.colors.Normalize(self.cmapLim[0], self.cmapLim[1])
        self.cmapVals = plt.cm.ScalarMappable(norm = self.cmapNorm, cmap = self.cmap)
        self.cmapVals.set_array([])
        # For colorbar mappable
        self.plot = self.cmapVals


    def plotFigure(self):
        for slice in self.listSlices2D:
            print('\nPlotting ' + self.name + '...')
            fColors = self.cmapVals.to_rgba(slice)
            print('\nfColors ready')
            self.axes[0].plot_surface(next(self.listX2D), next(self.listY2D), next(self.listZ2D), cstride = 1, rstride = 1, facecolors = fColors, vmin = self.cmapLim[0], vmax = self.cmapLim[1], shade = False)


    # def finalizeFigure(self, **kwargs):
    #     arZX = abs((self.zLim[1] - self.zLim[0])/(self.xLim[1] - self.xLim[0]))
    #     arYX = abs((self.yLim[1] - self.yLim[0])/(self.xLim[1] - self.xLim[0]))
    #     # Axes aspect ratio doesn't really work properly
    #     pbaspect = (1., arYX, arZX*2)
    #
    #     super(PlotSurfaceSlices3D, self).finalizeFigure(pbaspect = pbaspect, **kwargs)














if __name__ == '__main__':
    x = np.linspace(0, 300, 100)
    y = np.linspace(0, 100, 100)
    y2 = np.linspace(10, 80, 100)

    z2D = np.linspace(1, 10, x.size*y.size).reshape((y.size, x.size))
    z2D2 = np.linspace(10, 30, x.size*y.size).reshape((y.size, x.size))
    z2D3 = np.linspace(30, 60, x.size*y.size).reshape((y.size, x.size))

    # myplot = Plot2D_InsetZoom((x, x), (y, y2), z2D = (None,), zoomBox = (10, 60, 20, 40), save = True, equalAxis = True, figDir = 'R:/', name = 'newFig')

    # myplot = Plot2D_InsetZoom(x, y, z2D = z2D, zoomBox = (10, 70, 10, 30), save = True, equalAxis = True,
    #                           figDir = 'R:/', name = 'newFig2')

    # myplot = PlotSlices3D(x, y, [z2D, z2D2, z2D3], sliceOffsets = [0, 20, 50], name = '3d2', figDir = 'R:/', xLim = (0, 150), zDir = 'x')
    myplot = PlotContourSlices3D(x, y, [z2D, z2D2, z2D3], sliceOffsets = [20000, 20500, 21000], name = '3d2', figDir = 'R:/', zDir = 'x', xLabel = '$x$', yLabel = '$y$', zLabel = r'$z$ [m]', zLim = (0, 100), yLim = (0, 300), gradientBg = True)

    myplot.initializeFigure()

    myplot.plotFigure()

    myplot.finalizeFigure()


