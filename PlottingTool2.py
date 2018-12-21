import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
import numpy as np
from warnings import warn

class BaseFigure:
    def __init__(self, listX, listY, name = 'UntitledFigure', fontSize = 10, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, save = True, equalAxis = False, xLim = (None,), yLim = (None,), figWidth = 'half', subplots = (1, 1)):
        self.listX, self.listY = listX, listY
        self.name, self.figDir, self.save, self.show = name, figDir, save, show
        self.xLabel, self.yLabel, self.equalAxis = xLabel, yLabel, equalAxis
        self.xLim, self.yLim = xLim, yLim
        self.colors, self.gray = self.colors()
        self.subplots, self.figWidth, self.fontSize = subplots, figWidth, fontSize


    @staticmethod
    def colors(which = 'tableau10'):
        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        tableau = tableau20 if which is 'tableau20' else tableau10
        for i in range(len(tableau)):
            r, g, b = tableau[i]
            tableau[i] = (r/255., g/255., b/255.)

        tableauGray = (89/255., 89/255., 89/255.)
        return tableau, tableauGray


    @staticmethod
    def latexify(fig_width = None, fig_height = None, figWidth = 'halfpage', linewidth = 1, fontSize = 10, subplots = (1, 1)):
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
            fig_width = 3.39 if (figWidth is 'half') else 6.9  # inches

        if fig_height is None:
            golden_mean = (np.sqrt(5) - 1.0)/2.0  # Aesthetic ratio
            fig_height = fig_width*golden_mean  # height in inches
            fig_height *= 1.25*(subplots[0] - 1) if subplots[0] >1 else 1

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
            'legend.fontsize':     fontSize - 2,  # was 10
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
            "axes.spines.top":     False,
            "axes.spines.right" :  False,
            'axes.edgecolor':      tableauGray
            })


    def initializeFigure(self):
        self.latexify(fontSize = self.fontSize, figWidth = self.figWidth, subplots = self.subplots)
        self.fig, self.axes = plt.subplots(self.subplots[0], self.subplots[1], num = self.name)
        self.axes = (self.axes,) if not isinstance(self.axes, np.ndarray) else self.axes
        print('\n' + self.name + ' initialized')


    def plotFigure(self):
        print('\nPlotting ' + self.name + '...')
        if isinstance(self.listX, np.ndarray):
            self.listX, self.listY = (self.listX,), (self.listY,)


    def ensureMeshGrid(self):
        if len(np.array(self.listX[0]).shape) == 1:
            warn('\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid '
                    'automatically...\n',
                    stacklevel = 2)
            # Convert tuple to list
            self.listX, self.listY = list(self.listX), list(self.listY)
            self.listX[0], self.listY[0] = np.meshgrid(self.listX[0], self.listY[0], sparse = False)


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = True, setXYlabel = (True, True), grid = True):
        if len(self.listX) > 1:
            nCol = 2 if len(self.listX) > 3 else 1
            self.axes[0].legend(loc = 'best', shadow = False, fancybox = False, ncol = nCol)

        if grid:
            self.axes[0].grid(which = 'both', alpha = 0.25)

        if setXYlabel[0]:
            self.axes[0].set_xlabel(self.xLabel)

        if setXYlabel[1]:
            self.axes[0].set_ylabel(self.yLabel)

        if self.equalAxis:
            self.axes[0].set_aspect('equal', 'box')

        if self.xLim[0] is not None:
            self.axes[0].set_xlim(self.xLim)

        if self.yLim[0] is not None:
            self.axes[0].set_ylim(self.yLim)

        self.axes[0].set_xscale(xyScale[0]), self.axes[0].set_yscale(xyScale[1])
        if tightLayout:
            plt.tight_layout()

        print('\n' + self.name + ' finalized')
        if self.save:
            plt.savefig(self.figDir + '/' + self.name + '.png', transparent = True, bbox_inches = 'tight', dpi = 1000)
            print('\n' + self.name + '.png saved in ' + self.figDir)

        if self.show:
            plt.show()


class Plot2D(BaseFigure):
    def __init__(self, listX, listY, z2D = (None,), type = 'infer', alpha = 0.75, zLabel = '$z$', cmap = 'plasma', **kwargs):
        self.z2D = z2D
        if type is 'infer':
            if z2D[0] is not None:
                self.type = 'contourf'
            else:
                self.type = 'line'

        else:
            self.type = type
            # Don't know how to use it...
            assert type in ('scatter', 'contour')

        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'v', '^', '<', '>', 's', '8', 'p')*3
        self.alpha, self.cmap = alpha, cmap
        self.zLabel = zLabel

        super().__init__(listX, listY, **kwargs)


    def plotFigure(self, plotsLabel = (None,), contourLvl = 10):
        super().plotFigure()

        if self.type in ('contour', 'contourf'):
            self.ensureMeshGrid()

        self.plotsLabel = (self.type,)*len(self.listX) if plotsLabel[0] is None else self.plotsLabel
        self.plots = [None]*len(self.listX)
        for i in range(len(self.listX)):
            if self.type is 'line':
                self.plots[i] = self.axes[0].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = self.plotsLabel[i] + str(i + 1), color = self.colors[i], alpha = self.alpha)
            elif self.type is 'scatter':
                self.plots[i] = self.axes[0].scatter(self.listX[i], self.listY[i], lw = 0, label = self.plotsLabel[i] + str(i + 1), alpha = self.alpha, color = self.colors[i], marker = self.markers[i])
            elif self.type is 'contourf':
                self.plots[i] = self.axes[0].contourf(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            elif self.type is 'contour':
                self.plots[i] = self.axes[0].contour(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            else:
                warn("\nUnrecognized plot type! type must be one of ('infer', 'line', 'scatter', 'contourf', 'contour').\n", stacklevel = 2)
                return


    def finalizeFigure(self, cbarOrientate = 'horizontal', **kwargs):
        if self.type in ('contourf', 'contour') and len(self.axes) == 1:
            cb = plt.colorbar(orientation = cbarOrientate)
            cb.set_label(self.zLabel)
            cb.outline.set_visible(False)

        super().finalizeFigure(**kwargs)


class Plot2D_InsetZoom(Plot2D):
    def __init__(self, listX, listY, zoomBox, subplots = (2, 1), **kwargs):
        super().__init__(listX, listY, figWidth = 'full', subplots = subplots, **kwargs)
        self.zoomBox = zoomBox


    @staticmethod
    def mark_inset(parent_axes, inset_axes, loc1a = 1, loc1b = 1, loc2a = 2, loc2b = 2, **kwargs):
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
                self.axes[1].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = self.plotsLabel[i] + str(i + 1), alpha = self.alpha, color = self.colors[i])
            elif self.type is 'scatter':
                self.axes[1].scatter(self.listX[i], self.listY[i], lw = 0, label = self.plotsLabel[i] + str(i + 1), alpha = self.alpha, marker = self.markers[i])
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
        self.mark_inset(self.axes[0], self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none", ec = self.gray, ls = ':')
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

        plt.draw()
        # Single colorbar
        if self.type in ('contour', 'contourf'):
            self.fig.subplots_adjust(bottom = 0.1, top = 0.9, left = 0.1, right = 0.8)  # , wspace = 0.02, hspace = 0.2)
            cbar_ax = self.fig.add_axes((0.83, 0.1, 0.02, 0.8))
            cb = plt.colorbar(self.plots[0], cax = cbar_ax, orientation = 'vertical')
            cb.set_label(self.zLabel)
            cb.ax.tick_params(axis = 'y', direction = 'out')

        super().finalizeFigure(tightLayout = False, cbarOrientate = cbarOrientate, setXYlabel = setXYlabel, xyScale = xyScale, grid = False, **kwargs)


class PlotSlices3D(BaseFigure):
    def __init__(self, x2D, y2D, listSlices2D, sliceOffsets, zDir = 'z', zLabel = '$z$', contourLvl = 10, alpha = 1, viewAngles = (20, -120), zLim = (None,), cmap = 'plasma', cmapLabel = '$U$', **kwargs):
        super(PlotSlices3D, self).__init__(listX = (x2D,), listY = (y2D,), **kwargs)

        self.listSlices2D, self.sliceOffsets, self.zDir = iter(listSlices2D), iter(sliceOffsets), zDir
        self.zMin, self.zMax = (min(sliceOffsets), max(sliceOffsets)) if zLim[0] is None else (zLim[0], zLim[1])

        self.contourLvl, self.alpha, self.zLabel = contourLvl, alpha, zLabel
        self.cmap, self.cmapLabel = cmap, cmapLabel
        self.viewAngles = viewAngles
        # Multiplier to stretch the 'z' axis
        # When 1 slice, no stretch
        # When 3 slices, 1.3 in z, and 0.65 in other directions
        # self.figZ = 0.15*len(listSlices2D) + 0.85
        # self.figXY = -0.175*len(listSlices2D) + 1.175
        self.figZ = 1.
        self.figXY = 1
        self.cbarOrientate = 'vertical' if zDir is 'z' else 'horizontal'


    def initializeFigure(self):
        figSize = (2.5, 1) if self.zDir is 'z' else (1, 2.5)
        self.latexify(fontSize = self.fontSize, figWidth = self.figWidth, subplots = figSize)

        self.axes = (plt.figure(self.name).gca(projection = '3d'),)
        # self.axes[0].get_projection()
        if self.zDir is 'y':
            self.axes[0].get_proj = lambda: np.dot(Axes3D.get_proj(self.axes[0]), np.diag([self.figXY, self.figZ, self.figXY, 1]))
        elif self.zDir is 'x':
            self.axes[0].get_proj = lambda: np.dot(Axes3D.get_proj(self.axes[0]), np.diag([self.figZ, self.figXY, self.figXY, 1]))
        else:
            self.axes[0].get_proj = lambda: np.dot(Axes3D.get_proj(self.axes[0]), np.diag([self.figXY, self.figXY, self.figZ, 1]))


    def plotFigure(self):
        super(PlotSlices3D, self).plotFigure()

        self.ensureMeshGrid()
        # cmap = mpl.cm.get_cmap(self.cmap, 100)
        # bounds = np.arange(60)
        # vals = bounds[:-1]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for slice in self.listSlices2D:
            # self.plot = self.axes[0].contourf(self.listX[0], self.listY[0], slice, self.contourLvl, zdir = self.zDir, offset = next(self.sliceOffsets), alpha = self.alpha, cmap = self.cmap, norm = norm)
            self.plot = self.axes[0].contourf(self.listX[0], self.listY[0], slice, self.contourLvl, zdir = self.zDir,
                                              offset = next(self.sliceOffsets), alpha = self.alpha, cmap = self.cmap,
                                              levels = np.linspace(0, 80, 100))


    @staticmethod
    def format_3d_ax(ax):
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_yaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_zaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})


    def finalizeFigure(self, **kwargs):
        self.axes[0].set_zlabel(self.zLabel)
        self.axes[0].set_zlim(self.zMin, self.zMax)

        # bounds = np.arange(60)
        # vals = bounds[:-1]
        # cb = plt.colorbar(self.plot, orientation = self.cbarOrientate, extend = 'both', boundaries = bounds, values = vals, cmap = self.cmap)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(self.axes[0])
        # cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cb = plt.colorbar(self.plot, fraction=0.046, pad=0.04,orientation = self.cbarOrientate, extend = 'both')
        cb.set_label(self.cmapLabel)
        # Turn off background on all three panes
        self.format_3d_ax(self.axes[0])
        # self.axes[0].dist = 15
        # self.axes[0].w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # self.axes[0].w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # self.axes[0].w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.axes[0].view_init(self.viewAngles[0], self.viewAngles[1])

        super().finalizeFigure(grid = False, tightLayout = True, **kwargs)











if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    y2 = np.linspace(10, 80, 100)

    z2D = np.linspace(1, 10, x.size*y.size).reshape((y.size, x.size))
    z2D2 = np.linspace(10, 30, x.size*y.size).reshape((y.size, x.size))
    z2D3 = np.linspace(30, 60, x.size*y.size).reshape((y.size, x.size))

    # myplot = Plot2D_InsetZoom((x, x), (y, y2), z2D = (None,), zoomBox = (10, 60, 20, 40), save = True, equalAxis = True, figDir = 'R:/', name = 'newFig')

    # myplot = Plot2D_InsetZoom(x, y, z2D = z2D, zoomBox = (10, 70, 10, 30), save = True, equalAxis = True,
    #                           figDir = 'R:/', name = 'newFig2')

    myplot = PlotSlices3D(x, y, [z2D, z2D2, z2D3], sliceOffsets = [0, 20, 50], name = '3d2', figDir = 'R:/')

    myplot.initializeFigure()

    myplot.plotFigure()

    myplot.finalizeFigure()


