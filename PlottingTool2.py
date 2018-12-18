import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
import numpy as np
from warnings import warn

class BaseFigure:
    def __init__(self, name = 'UntitledFigure', fontSize = 14, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,)):
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
            gridspec_kw = {'wspace': 0, 'hspace': 0}
        else:
            gridspec_kw = {'wspace': None, 'hspace': None}

        self.fig, self.axes = plt.subplots(nrow, ncol, num = self.name, gridspec_kw = gridspec_kw)
        if not isinstance(self.axes, np.ndarray):
            self.axes = (self.axes,)

        print('\n' + self.name + ' initialized')


    def plotFigure(self, listX, listY):
        print('\nPlotting ' + self.name + '...')
        if isinstance(listX, np.ndarray):
            listX, listY = (listX,), (listY,)

        return listX, listY


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = True):
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
    def __init__(self, listX, listY, alpha = 1, name = 'UntitledFigure', fontSize = 14, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,)):
        super().__init__(name, fontSize, xLabel, yLabel, figDir, show, saveFig, equalAxis, useTex, linewidth, xLim, yLim)
        self.listX, self.listY = listX, listY
        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'v', '^', '<', '>', 's', '8', 'p')*3
        self.alpha = alpha


    def plotFigure(self, type = 'line', plotLabels = (None,)):
        self.listX, self.listY = super().plotFigure(self.listX, self.listY)
        if plotLabels[0] is None:
            self.plotLabels = (type,)*len(self.listX)

        self.plots = [None]*len(self.listX)
        for i in range(len(self.listX)):
            if type is 'line':
                self.plots[i] = self.axes[0].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = self.plotLabels[i] + str(i + 1), marker = self.markers[i], alpha = self.alpha)
            else:
                self.plots[i] = self.axes[0].scatter(self.listX[i], self.listY[i], lw = 0, label = self.plotLabels[i] + str(i + 1), alpha = self.alpha)


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = True):
        if len(self.listX) > 1:
            if len(self.listX) > 3:
                nCol = 2
            else:
                nCol = 1
            plt.legend(loc = 'best', shadow = True, fancybox = False, ncol = nCol)
            plt.grid(which = 'both', alpha = 0.5)

        super().finalizeFigure(xyScale, tightLayout)






class Plot2D_InsetZoom(Plot2D):
    def __init__(self, listX, listY, zoomBox, alpha = 1, name = 'UntitledFigure', fontSize = 14, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,)):
        super().__init__(listX, listY, alpha, name, fontSize, xLabel, yLabel, figDir, show, saveFig, equalAxis, useTex, linewidth, xLim, yLim)
        self.zoomBox = zoomBox


    def mark_inset(self, parent_axes, inset_axes, loc1a = 1, loc1b = 1, loc2a = 2, loc2b = 2, **kwargs):
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


    def plotFigure(self, type = 'line', plotLabels = (None,)):
        super().plotFigure(type, plotLabels)
        for i in range(len(self.listX)):
            if type is 'line':
                self.axes[1].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = self.plotLabels[i] + str(i + 1), marker = self.markers[i], alpha = self.alpha)
            else:
                self.axes[1].scatter(self.listX[i], self.listY[i], lw = 0, label = self.plotLabels[i] + str(i + 1), alpha = self.alpha)


    def finalizeFigure(self, xyScale = ('linear', 'linear'), tightLayout = False):
        self.axes[1].set_xlim(self.zoomBox[0], self.zoomBox[1]), self.axes[1].set_ylim(self.zoomBox[2], self.zoomBox[3])
        self.axes[1].set_xlabel(self.xLabel), self.axes[1].set_ylabel(self.yLabel)
        if self.equalAxis:
            self.axes[1].set_aspect('equal', 'box')

        self.axes[1].set_xscale(xyScale[0]), self.axes[1].set_yscale(xyScale[1])
        self.mark_inset(self.axes[0], self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none", ec = "gray", ls = ':')
        # self.axes[0].spines['top'].set_visible(False), self.axes[0].spines['right'].set_visible(False)
        # self.axes[0].spines['bottom'].set_visible(False), self.axes[0].spines['left'].set_visible(False)
        self.axes[1].spines['bottom'].set_color('gray'), self.axes[1].spines['top'].set_color('gray')
        self.axes[1].spines['right'].set_color('gray'), self.axes[1].spines['left'].set_color('gray')
        self.axes[1].spines['bottom'].set_linestyle(':'), self.axes[1].spines['top'].set_linestyle(':')
        self.axes[1].spines['right'].set_linestyle(':'), self.axes[1].spines['left'].set_linestyle(':')
        plt.draw()

        saveFigState, showState = self.saveFig, self.show
        self.saveFig, self.show = False, False
        super().finalizeFigure(xyScale, tightLayout)

        plt.tight_layout()
        self.saveFig, self.show = saveFigState, showState
        if self.saveFig:
            plt.savefig(self.figDir + '/' + self.name + '.png', transparent = True, bbox_inches = 'tight', dpi = 1000)

        if self.show:
            plt.show()










if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)

    myplot = Plot2D_InsetZoom(x, y, zoomBox = (10, 40, 20, 40), saveFig = False)

    myplot.initializeFigure(nrow = 2)

    myplot.plotFigure()

    myplot.finalizeFigure()


