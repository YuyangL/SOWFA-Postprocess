import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
import numpy as np
from warnings import warn

class BaseFigure:
    def __init__(self, listX, listY, name = 'UntitledFigure', fontSize = 14, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,)):
        self.listX, self.listY = listX, listY
        self.name, self.figDir, self.saveFig, self.show = name, figDir, saveFig, show
        self.xLabel, self.yLabel, self.equalAxis = xLabel, yLabel, equalAxis
        self.xLim, self.yLim = xLim, yLim
        # For contour/contourf plots with inset zoom only
        self.contractSubplots = False
        mpl.rcParams.update({   "legend.framealpha": 0.75,
                                'font.size':         fontSize,
                                'text.usetex':       useTex,
                                'font.family':       'serif',
                                'lines.linewidth':   linewidth})


    def initializeFigure(self, nrow = 1, ncol = 1):
        if self.contractSubplots:
            gridspec_kw = {'wspace': None, 'hspace': None}
        else:
            gridspec_kw = {'wspace': None, 'hspace': None}

        self.fig, self.axes = plt.subplots(nrow, ncol, num = self.name, gridspec_kw = gridspec_kw)
        if not isinstance(self.axes, np.ndarray):
            self.axes = (self.axes,)

        print('\n' + self.name + ' initialized')


    def plotFigure(self):
        print('\nPlotting ' + self.name + '...')
        if isinstance(self.listX, np.ndarray):
            self.listX, self.listY = (self.listX,), (self.listY,)


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = True):
        if len(self.listX) > 1:
            if len(self.listX) > 3:
                nCol = 2
            else:
                nCol = 1
            plt.legend(loc = 'best', shadow = True, fancybox = False, ncol = nCol)
            plt.grid(which = 'both', alpha = 0.5)

        # for ax in self.axes:
        self.axes[0].set_xlabel(self.xLabel), self.axes[0].set_ylabel(self.yLabel)
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
        if self.saveFig:
            plt.savefig(self.figDir + '/' + self.name + '.png', transparent = True, bbox_inches = 'tight', dpi = 1000)
            print('\n' + self.name + '.png saved in ' + self.figDir)

        if self.show:
            plt.show()


class Plot2D(BaseFigure):
    def __init__(self, listX, listY, z2D = (None,), type = 'infer', alpha = 1, name = 'UntitledFigure', fontSize = 14, xLabel = '$x$', yLabel = '$y$', zLabel = '$z$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,), cmap = 'plasma'):
        super().__init__(listX, listY, name, fontSize, xLabel, yLabel, figDir, show, saveFig, equalAxis, useTex, linewidth, xLim, yLim)
        self.z2D = z2D
        if type is 'infer':
            if z2D[0] is not None:
                self.type = 'contourf'
                self.contractSubplots = True
            else:
                self.type = 'line'

        else:
            self.type = type
            # Don't know how to use it...
            assert type in ('scatter', 'contour')

        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'v', '^', '<', '>', 's', '8', 'p')*3
        self.alpha, self.cmap = alpha, cmap
        self.zLabel = zLabel


    def plotFigure(self, plotsLabel = (None,), contourLvl = 10):
        super().plotFigure()
        if self.type in ('contour', 'contourf'):
            if len(np.array(self.listX[0]).shape) == 1:
                warn('\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid automatically...\n', stacklevel = 2)
                # Convert tuple to list
                self.listX, self.listY = list(self.listX), list(self.listY)
                self.listX[0], self.listY[0] = np.meshgrid(self.listX[0], self.listY[0], sparse = False)

        if plotsLabel[0] is None:
            self.plotsLabel = (type,)*len(self.listX)

        self.plots = [None]*len(self.listX)
        for i in range(len(self.listX)):
            if self.type is 'line':
                self.plots[i] = self.axes[0].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = self.plotsLabel[i] + str(i + 1), marker = self.markers[i], alpha = self.alpha)
            elif self.type is 'scatter':
                self.plots[i] = self.axes[0].scatter(self.listX[i], self.listY[i], lw = 0, label = self.plotsLabel[i] + str(i + 1), alpha = self.alpha)
            elif self.type is 'contourf':
                self.plots[i] = self.axes[0].contourf(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            elif self.type is 'contour':
                self.plots[i] = self.axes[0].contour(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            else:
                warn("\nUnrecognized plot type! type must be one of ('infer', 'line', 'scatter', 'contourf', 'contour').\n", stacklevel = 2)
                return


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = True, cbarOrientate = 'horizontal'):
        if self.type in ('contourf', 'contour') and len(self.axes) == 1:
            cb = plt.colorbar(orientation = cbarOrientate)
            cb.set_label(self.zLabel)
            cb.outline.set_visible(False)

        super().finalizeFigure(xyScale, tightLayout)


class Plot2D_InsetZoom(Plot2D):
    def __init__(self, listX, listY, zoomBox, z2D = (None,), type = 'infer', alpha = 1, name = 'UntitledFigure', fontSize = 10, xLabel = '$x$', yLabel = '$y$', zLabel = '$z$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,), cmap = 'plasma'):
        super().__init__(listX, listY, z2D, type, alpha, name, fontSize, xLabel, yLabel, zLabel, figDir, show, saveFig, equalAxis, useTex, linewidth, xLim, yLim, cmap)
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
            if type is 'line':
                self.axes[1].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = self.plotsLabel[i] + str(i + 1), marker = self.markers[i], alpha = self.alpha)
            elif self.type is 'scatter':
                self.axes[1].scatter(self.listX[i], self.listY[i], lw = 0, label = self.plotsLabel[i] + str(i + 1), alpha = self.alpha)
            elif self.type is 'contourf':
                self.axes[1].contourf(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')
            elif self.type is 'contour':
                self.axes[1].contour(self.listX[i], self.listY[i], self.z2D, levels = contourLvl, cmap = self.cmap, extend = 'both')


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = False, cbarOrientate = 'vertical'):
        self.axes[1].set_xlim(self.zoomBox[0], self.zoomBox[1]), self.axes[1].set_ylim(self.zoomBox[2], self.zoomBox[3])
        self.axes[1].set_xlabel(self.xLabel), self.axes[1].set_ylabel(self.yLabel)
        if self.equalAxis:
            self.axes[1].set_aspect('equal', 'box')

        self.axes[1].set_xscale(xyScale[0]), self.axes[1].set_yscale(xyScale[1])
        self.mark_inset(self.axes[0], self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none", ec = "gray", ls = ':')
        if self.type in ('contour', 'contourf'):
            self.axes[0].spines['top'].set_visible(False), self.axes[0].spines['right'].set_visible(False)
            self.axes[0].spines['bottom'].set_visible(False), self.axes[0].spines['left'].set_visible(False)

        self.axes[1].spines['bottom'].set_color('gray'), self.axes[1].spines['top'].set_color('gray')
        self.axes[1].spines['right'].set_color('gray'), self.axes[1].spines['left'].set_color('gray')
        self.axes[1].spines['bottom'].set_linestyle(':'), self.axes[1].spines['top'].set_linestyle(':')
        self.axes[1].spines['right'].set_linestyle(':'), self.axes[1].spines['left'].set_linestyle(':')
        plt.draw()

        saveFigState, showState = self.saveFig, self.show
        self.saveFig, self.show = False, False
        super().finalizeFigure(xyScale, tightLayout, cbarOrientate = 'vertical')

        plt.tight_layout()

        if self.type in ('contour', 'contourf'):
            # self.fig.subplots_adjust(right = 0.8)
            # cbar_ax = self.fig.add_axes([0.85, 0.15, 0.02, 0.7])
            # cb = plt.colorbar(self.plots[0], cax = cbar_ax, orientation = 'vertical')
            divider = make_axes_locatable(self.fig.gca())
            cax = divider.append_axes("right", "5%", pad = "3%")
            cb = plt.colorbar(self.plots[0], cax = cax, orientation = 'vertical')

            cb.set_label(self.zLabel)
            cb.outline.set_visible(False)

            plt.tight_layout()

        self.saveFig, self.show = saveFigState, showState
        if self.saveFig:
            plt.savefig(self.figDir + '/' + self.name + '.png', transparent = True, bbox_inches = 'tight', dpi = 1000)

        if self.show:
            plt.show()










if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 60, 100)

    z2D = np.linspace(1, 20, x.size*y.size).reshape((y.size, x.size))

    myplot = Plot2D_InsetZoom(x, y, z2D = z2D, zoomBox = (10, 40, 20, 40), saveFig = False, equalAxis = True)

    myplot.initializeFigure(nrow = 2)

    myplot.plotFigure()

    myplot.finalizeFigure()


