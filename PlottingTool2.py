ahfifboaubfob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from warnings import warn

class BaseFigure:
    def __init__(self, name = 'UntitledFigure', fontSize = 14, xLabel = '$x$', yLabel = '$y$', figDir = './', show = True, saveFig = True, equalAxis = False, useTex = True, linewidth = 1, xLim = (None,), yLim = (None,)):
        self.name, self.figDir, self.saveFig, self.show = name, figDir, saveFig, show
        self.xLabel, self.yLabel, self.equalAxis = xLabel, yLabel, equalAxis
        self.xLim, self.yLim = xLim, yLim
        mpl.rcParams.update({   "legend.framealpha": 0.75,
                                'font.size':         fontSize,
                                'text.usetex':       useTex,
                                'font.family':       'serif',
                                'lines.linewidth':   linewidth})


    def initializeFigure(self, nrow = 1, ncol = 1):
        self.fig, self.axes = plt.subplots(nrow, ncol, num = self.name)
        if not isinstance(self.axes, tuple):
            self.axes = (self.axes,)

        print('\n' + self.name + ' initialized')


    def plotFigure(self, listX, listY):
        print('\nPlotting ' + self.name + '...')
        if isinstance(listX, np.ndarray):
            listX, listY = (listX,), (listY,)
        # else:
        #     listX, listY = listX, listY

        return listX, listY


    def finalizeFigure(self, xScale = 'linear', yScale = 'linear'):
        for ax in self.axes:
            ax.set_xlabel(self.xLabel), ax.set_ylabel(self.yLabel)
            if self.equalAxis:
                ax.set_aspect('equal', 'box')

            if self.xLim[0] is not None:
                ax.set_xlim(self.xLim)

            if self.yLim[0] is not None:
                ax.set_ylim(self.yLim)

            ax.set_xscale(xScale), ax.set_yscale(yScale)

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
            plotLabels = (type,)*len(self.listX)

        for i in range(len(self.listX)):
            if type is 'line':
                self.axes[0].plot(self.listX[i], self.listY[i], ls = self.lines[i], label = plotLabels[i] + str(i + 1), marker = self.markers[i], alpha = self.alpha)
            else:
                self.axes[0].scatter(self.listX[i], self.listY[i], lw = 0, label = plotLabels[i] + str(i + 1), alpha = self.alpha)


    def finalizeFigure(self, xScale = 'linear', yScale = 'linear'):
        if len(self.listX) > 1:
            if len(self.listX) > 3:
                nCol = 2
            else:
                nCol = 1
            plt.legend(loc = 'best', shadow = True, fancybox = False, ncol = nCol)
            plt.grid(which = 'both', alpha = 0.5)

        super().finalizeFigure(xScale, yScale)




if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)

    myplot = Plot2D(x, y, saveFig = False)

    myplot.initializeFigure()

    myplot.plotFigure()

    myplot.finalizeFigure()


