import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from warnings import warn
from Utilities import timer
from numba import njit, jit, prange
from scipy import ndimage

class BaseFigure:
    def __init__(self, list_x, list_y, name='UntitledFigure', fontsize=8, xlabel='$x$', ylabel='$y$', figdir='./', 
                 show=True, save=True, equalaxis=False, xlim=None, ylim=None,
                 figwidth='half', figheight_multiplier=1., colors='tableau10', font='Utopia', **kwargs):
        # Ensure tuple of arrays
        self.list_x, self.list_y = ((list_x,), (list_y,)) if isinstance(list_x, np.ndarray) else (list_x, list_y)
        # Number of provided arrays
        self.narr = len(self.list_x)
        self.name, self.figdir, self.save, self.show = name, figdir, save, show
        if not self.show:
            plt.ioff()
        else:
            plt.ion()
        
        self.xlabel, self.ylabel = xlabel, ylabel
        self.equalaxis = equalaxis
        self.xlim, self.ylim = xlim, ylim
        self.colors, self.gray = self.getColors(which=colors) if colors in ('tableau10', 'tableau20', 'qualitative') else (colors, (89/255., 89/255., 89/255.))
        # Figure related variables
        self.figwidth, self.figheight_multiplier, self.fontsize = figwidth, figheight_multiplier, fontsize
        self.font = font


    @staticmethod
    def getColors(which='qualitative'):
        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (23, 190, 207), (214, 39, 40), (188, 189, 34), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)]
        # Orange, blue, magenta, cyan, red, teal, grey
        qualitative = [(238, 119, 51), (0, 119, 187), (238, 51, 119), (51, 187, 238), (204, 51, 117), (0, 153, 136), (187, 187, 187)]
        colorsdict = {'tableau20': tableau20,
                      'tableau10': tableau10,
                      'qualitative': qualitative}
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        colors = colorsdict[which]
        for i in range(len(colors)):
            r, g, b = colors[i]
            colors[i] = (r/255., g/255., b/255.)

        gray = (89/255., 89/255., 89/255.)
        
        return colors, gray


    def _latexify(self, fig_width=None, fig_height=None, figspan='half', linewidth=1, fontsize=8, subplots=(1, 1), figheight_multiplier=1.,
                  **kwargs):
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
                if figspan == '1/3':
                    fig_width = 2.3
                elif figspan == 'half':
                    fig_width = 3.39  # inches
                else:
                    fig_width = 6.9
            else:
                fig_width = 6.9  # inches

        if fig_height is None:
            golden_mean = (np.sqrt(5) - 1.0)/2.0  # Aesthetic ratio
            # In case subplots option is not applicable e.g. normal Plot2D and you still want elongated height
            fig_height = fig_width*golden_mean*figheight_multiplier  # height in inches
            fig_height *= subplots[0]

        MAX_HEIGHT_INCHES = 8.0
        if fig_height > MAX_HEIGHT_INCHES:
            warn("\nfig_height too large:" + str(fig_height) +
                  ". Will reduce to " + str(MAX_HEIGHT_INCHES) + " inches", stacklevel = 2)
            fig_height = MAX_HEIGHT_INCHES

        mpl.rcParams.update({
            'backend':             'Qt5Agg',
            'text.latex.preamble': [r"\usepackage{gensymb,amsmath}"],
            'axes.labelsize':      fontsize - 2.,  # fontsize for x and y labels (was 10)
            'axes.titlesize':      fontsize,
            'font.size':           fontsize - 2.,  # was 10
            'legend.fontsize':     fontsize - 4.,  # was 10
            'xtick.labelsize':     fontsize - 4.,
            'ytick.labelsize':     fontsize - 4.,
            'xtick.color':         self.gray,
            'ytick.color':         self.gray,
            'xtick.direction':     'out',
            'ytick.direction':     'out',
            'text.usetex':         True,
            'figure.figsize':      (fig_width, fig_height),
            'font.family':         'serif',
            # 'font.serif':          self.font,
            "legend.framealpha":   0.75,
            'legend.edgecolor':    'none', #'none'
            'lines.linewidth':     linewidth,
            'lines.markersize':    2,
            'lines.markeredgewidth': 0,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            'axes.edgecolor':      self.gray,
            'axes.formatter.limits': (-4, 5),
            'lines.antialiased':   True,
            'patch.antialiased':   True,
            'text.antialiased':    True})


    def initializeFigure(self, constrained_layout=True, **kwargs):
        self._latexify(fontsize=self.fontsize, figspan=self.figwidth, figheight_multiplier=self.figheight_multiplier, **kwargs)

        self.fig, self.axes = plt.subplots(num=self.name, constrained_layout=constrained_layout)
        print('\nFigure ' + self.name + ' initialized')


    def plotFigure(self, **kwargs):
        print('\nPlotting ' + self.name + '...')


    def _ensureMeshGrid(self):
        if len(np.array(self.list_x[0]).shape) == 1:
            warn('\nX and Y are 1D, contour/contourf requires mesh grid. Converting X and Y to mesh grid '
                    'automatically...\n',
                    stacklevel = 2)
            # Convert tuple to list
            self.list_x, self.list_y = list(self.list_x), list(self.list_y)
            self.list_x[0], self.list_y[0] = np.meshgrid(self.list_x[0], self.list_y[0], sparse=False)


    def finalizeFigure(self, xyscale=('linear', 'linear'), show_xylabel=(True, True), grid=True,
                       transparent_bg=False, legloc='best', showleg=True,
                       tight_layout=False):
        if len(self.list_x) > 1 and showleg:
            ncol = 2 if len(self.list_x) > 3 else 1
            self.axes.legend(loc=legloc, shadow=False, fancybox=False, ncol=ncol)

        if grid: self.axes.grid(which='major', alpha=0.25)

        if show_xylabel[0]: self.axes.set_xlabel(self.xlabel)
        if show_xylabel[1]: self.axes.set_ylabel(self.ylabel)

        # Only when x, y scales are linear can equal axis take effect
        if self.equalaxis and xyscale[0] == 'linear' and xyscale[1] == 'linear':
            # Only execute 2D equal axis if the figure is actually 2D
            try:
                self.viewangle
            except AttributeError:
                self.axes.set_aspect('equal', 'box')

        if self.xlim is not None: self.axes.set_xlim(self.xlim)
        if self.ylim is not None: self.axes.set_ylim(self.ylim)

        if xyscale[0] in ('linear', 'symlog', 'log'): self.axes.set_xscale(xyscale[0])
        if xyscale[1] in ('linear', 'symlog', 'log'): self.axes.set_yscale(xyscale[1])
        if tight_layout: plt.tight_layout()

        print('\nFigure ' + self.name + ' finalized')
        if self.save:
            plt.savefig(self.figdir + '/' + self.name + '.png', transparent=transparent_bg,
                        dpi=1000)
            print('\nFigure ' + self.name + '.png saved in ' + self.figdir)

        # Close current figure window
        # so that the next figure will be based on a new figure window even if the same name 
        plt.show() if self.show else plt.close()       


class Plot2D_Image(BaseFigure):
    def __init__(self, val, extent=None, val_label='$z$', cmap='plasma', val_lim=None, rotate_img=True, **kwargs):
        self.rotate_img = rotate_img
        if rotate_img:
            val = ndimage.rotate(val, 90) if len(val.shape) >= 3 else val.T

        self.val = val
        self.val_label = val_label
        # Ensure there're two entries, one for vmin, one for vmax
        self.val_lim = (None, None) if val_lim is None else val_lim  
        self.cmap = cmap
        self.extent = extent
        super().__init__(list_x=(None,), list_y=(None,), **kwargs)
        
        
    def initializeFigure(self, **kwargs):
        super(Plot2D_Image, self).initializeFigure(**kwargs)


    def plotFigure(self, origin='infer', norm=None, **kwargs):
        if origin == 'infer':
            self.origin = 'lower' if self.rotate_img and len(self.val.shape) == 2 else 'upper'

        if norm in ('symlog', 'SymLog'):
            self.norm=SymLogNorm(linthresh=0.03)
        elif norm in ('log', 'Log'):
            self.norm=LogNorm()
        else:
            self.norm=None
            
        self.plots = self.axes.imshow(self.val, origin=self.origin, aspect='equal', extent=self.extent, cmap=self.cmap, vmin=self.val_lim[0], vmax=self.val_lim[1], norm=self.norm)


    def finalizeFigure(self, cbar_orient='horizontal', showcb=True, cbticks=None, **kwargs):
        if showcb:
            cb = plt.colorbar(self.plots, ax=self.axes, orientation=cbar_orient, extend='both', ticks=cbticks)
            cb.set_label(self.val_label)

        super().finalizeFigure(grid=False, xyscale=(None, None), **kwargs)


class Plot2D(BaseFigure):
    def __init__(self, list_x, list_y, val=None, val_lim=None, 
                 plot_type='infer', alpha=0.75, val_label='$z$', cmap='plasma',
                 grad_bg=False, grad_bg_range=None, grad_bg_dir='x', **kwargs):
        self.val = val
        # Ensure there's an entry for both vmin and vmax
        self.val_lim = val_lim if val_lim is not None else (None, None)
        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'D', 's', 'X', 'v', '^', '<', '>', 's', '8', 'p')*3
        self.alpha, self.cmap = alpha, cmap
        self.val_label = val_label
        # Gradient background related variables
        self.grad_bg, self.grad_bg_dir = grad_bg, grad_bg_dir
        # Ensure there's an entry for both vmin and vmax
        self.grad_bg_range = (None, None) if grad_bg_range is None else grad_bg_range

        super().__init__(list_x, list_y, **kwargs)
        
        # If gradient background is enabled, there has to have x and y limits
        if grad_bg:
            if self.xlim is None:
                self.xlim = np.min(self.list_x), np.max(self.list_x)
            
            if self.ylim is None:
                self.ylim = np.min(self.list_y), np.max(self.list_y)

        # If multiple data provided, make sure plot_type is a tuple of the same length
        if plot_type == 'infer':
            self.plot_type = ('contourf',)*self.narr if val is not None else ('line',)*self.narr
        else:
            self.plot_type = (plot_type,)*self.narr if isinstance(plot_type, str) else plot_type
            
    
    def initializeFigure(self, **kwargs):
        super(Plot2D, self).initializeFigure(**kwargs)


    def plotFigure(self, linelabel=None, showmarker=False, contour_lvl=20, **kwargs):
        self.showmarker, self.contour_lvl = showmarker, contour_lvl
        # Gradient background, only for line and scatter plots
        if self.grad_bg and self.plot_type[0] in ('line', 'scatter'):
            xmesh, ymesh = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.linspace(self.ylim[0], self.ylim[1], 3))
            graymesh = (np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.arange(3)))[0] if self.grad_bg_dir is 'x' else (np.meshgrid(np.arange(3), np.linspace(self.ylim[0], self.ylim[1], 3)))[1]
            self.axes[0].contourf(xmesh, ymesh, graymesh, 500, cmap='gray', alpha=0.333, vmin=self.grad_bg_range[0], vmax=self.grad_bg_range[1])

        super().plotFigure(**kwargs)

        self.linelabel = np.arange(1, len(self.list_x) + 1) if linelabel is None else linelabel
        self.plots = [None]*self.narr
        for i in range(self.narr):
            if self.plot_type[i] == 'line':
                if not showmarker:
                    self.plots[i] = self.axes.plot(self.list_x[i], self.list_y[i], ls=self.lines[i], label=str(self.linelabel[i]), color=self.colors[i], alpha=self.alpha)
                else:
                    self.plots[i] = self.axes.plot(self.list_x[i], self.list_y[i], ls=self.lines[i],
                                                   label=str(self.linelabel[i]), color=self.colors[i], alpha=self.alpha, 
                                                   marker=self.markers[i])
                    
            elif self.plot_type[i] == 'scatter':
                self.plots[i] = self.axes.scatter(self.list_x[i], self.list_y[i], lw=0, label=str(self.linelabel[i]), alpha=self.alpha, color=self.colors[i], marker=self.markers[i])
            elif self.plot_type[i] == 'contourf':
                self._ensureMeshGrid()
                self.plots[i] = self.axes.contourf(self.list_x[i], self.list_y[i], self.val, levels=contour_lvl, cmap=self.cmap, extend='both', vmin=self.val_lim[0], vmax=self.val_lim[1])
            elif self.plot_type[i] == 'contour':
                self._ensureMeshGrid()
                self.plots[i] = self.axes.contour(self.list_x[i], self.list_y[i], self.val, levels=contour_lvl, cmap=self.cmap, extend='both',
                                                     vmin=self.val_lim[0], vmax=self.val_lim[1])
            else:
                warn("\nUnrecognized plot plot_type! plot_type must be one/list of ('infer', 'line', 'scatter', 'contourf', 'contour').\n", stacklevel=2)
                return


    def finalizeFigure(self, cbar_orient='horizontal', showcb=True, cbticks=None, **kwargs):
        if self.plot_type[0] in ('contourf', 'contour'):
            cb = plt.colorbar(self.plots[0], ax=self.axes, orientation=cbar_orient, extend='both', ticks=cbticks)
            cb.set_label(self.val_label)
            super().finalizeFigure(grid=False, **kwargs)
        else:
            super().finalizeFigure(**kwargs)


# TODO: verify
class Plot2D_InsetZoom(Plot2D):
    def __init__(self, list_x, list_y, zoomBox, subplots = (2, 1), **kwargs):
        super().__init__(list_x, list_y, figwidth = 'full', subplots = subplots, **kwargs)

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


    def plotFigure(self, linelabel = (None,), contour_lvl = 10, **kwargs):
        super().plotFigure(linelabel, contour_lvl)
        for i in range(len(self.list_x)):
            if self.plot_type is 'line':
                self.axes[1].plot(self.list_x[i], self.list_y[i], ls = self.lines[i], label = str(self.linelabel[i]), alpha = self.alpha, color = self.colors[i])
            elif self.plot_type is 'scatter':
                self.axes[1].scatter(self.list_x[i], self.list_y[i], lw = 0, label = str(self.linelabel[i]), alpha = self.alpha, marker = self.markers[i])
            elif self.plot_type is 'contourf':
                self.axes[1].contourf(self.list_x[i], self.list_y[i], self.val, levels = contour_lvl, cmap = self.cmap, extend = 'both')
            elif self.plot_type is 'contour':
                self.axes[1].contour(self.list_x[i], self.list_y[i], self.val, levels = contour_lvl, cmap = self.cmap, extend = 'both')


    def finalizeFigure(self, cbar_orient = 'vertical', show_xylabel = (False, True), xyscale = ('linear', 'linear'), **kwargs):
        self.axes[1].set_xlim(self.zoomBox[0], self.zoomBox[1]), self.axes[1].set_ylim(self.zoomBox[2], self.zoomBox[3])
        self.axes[1].set_xlabel(self.xlabel), self.axes[1].set_ylabel(self.ylabel)
        if self.equalaxis:
            self.axes[1].set_aspect('equal', 'box')

        self.axes[1].set_xscale(xyscale[0]), self.axes[1].set_yscale(xyscale[1])
        self._mark_inset(self.axes[0], self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none",
                         ec = self.gray, ls = ':')
        if self.plot_type in ('contour', 'contourf'):
            for ax in self.axes:
                ax.tick_params(axis = 'both', direction = 'out')

        else:
            self.axes[1].grid(which = 'both', alpha = 0.25)
            if len(self.list_x) > 1:
                ncol = 2 if len(self.list_x) > 3 else 1
                self.axes[1].legend(loc = 'best', shadow = False, fancybox = False, ncol = ncol)

        for spine in ('top', 'bottom', 'left', 'right'):
            if self.plot_type in ('contour', 'contourf'):
                self.axes[0].spines[spine].set_visible(False)
            self.axes[1].spines[spine].set_visible(True)
            self.axes[1].spines[spine].set_linestyle(':')

        # plt.draw()
        # Single colorbar
        if self.plot_type in ('contour', 'contourf'):
            self.fig.subplots_adjust(bottom = 0.1, top = 0.9, left = 0.1, right = 0.8)  # , wspace = 0.02, hspace = 0.2)
            cbar_ax = self.fig.add_axes((0.83, 0.1, 0.02, 0.8))
            cb = plt.colorbar(self.plots[0], cax = cbar_ax, orientation = 'vertical')
            cb.set_label(self.zlabel)
            cb.ax.tick_params(axis = 'y', direction = 'out')

        super().finalizeFigure(cbar_orient = cbar_orient, show_xylabel = show_xylabel, xyscale = xyscale, grid = False, **kwargs)


class Plot2D_MultiAxes(Plot2D):
    def __init__(self, list_x, list_y, list_x2, list_y2, ax2label='$x_2$', 
                 plot_type2='line', ax2loc='x',
                 x2lim=None, y2lim=None, **kwargs):
        super(Plot2D_MultiAxes, self).__init__(list_x, list_y, equalaxis=False, **kwargs)
        
        self.list_x2, self.list_y2 = list_x2, list_y2
        # Ensure tuple
        if isinstance(self.list_x2, np.ndarray): self.list_x2 = (self.list_x2,)
        if isinstance(self.list_y2, np.ndarray): self.list_y2 = (self.list_y2,)
        self.narr2 = len(self.list_x2)
        # Whether new axes are on x axis or y axis
        self.ax2loc = ax2loc
        # The type of plots in x, y provided in original list 1 and new list 2 respectively
        self.plot_type2 = (plot_type2,)*self.narr2 if isinstance(plot_type2, str) else plot_type2
            
        # Ensure ax2 labels are the same length of provided x2 and y2 lists
        if isinstance(ax2label, str):
            self.ax2label = (ax2label,)*self.narr2
            
        self.x2lim = x2lim
        self.y2lim = y2lim


    def initializeFigure(self, ymargin='auto', xmargin='auto', **kwargs):
        # constrained_layout not support by subplots_adjust()
        super(Plot2D_MultiAxes, self).initializeFigure(constrained_layout=False, **kwargs)

        # Empirical relation to scale margin with number of provided extra x2 and/or y2
        self.ymargin = 0.75**self.narr2 if ymargin == 'auto' else ymargin
        self.xmargin = 0.25 + (self.narr2 - 2)*0.1 if xmargin == 'auto' else xmargin

        self.fig.subplots_adjust(right=self.ymargin) if self.ax2loc == 'y' else self.fig.subplots_adjust(bottom=self.xmargin)
        # Set host patch to transparent so that 2nd axis plot won't be hindered by the white background.
        # If save as pdf then 2nd axis plot is not hindered regardless
        self.axes.patch.set_visible(False)


    def plotFigure(self, linelabel2=None, xyscale2=('linear', 'linear'), **kwargs):
        self.linelabel2 = np.arange(1, self.narr2 + 1) if linelabel2 is None else linelabel2

        super(Plot2D_MultiAxes, self).plotFigure(**kwargs)
                
        self.axes2, self.plots2 = ([None]*self.narr2,)*2
        for i in range(self.narr2):
            color = self.colors[i + self.narr] if self.plot_type2[i] != 'shade' else (160/255., 160/255., 160/255.)
            # If new axes are x
            if self.ax2loc == 'x':
                self.axes2[i] = self.axes.twiny()
                # Move twinned axis ticks and label from top to bottom
                self.axes2[i].xaxis.set_ticks_position("bottom")
                self.axes2[i].xaxis.set_label_position("bottom")
                # Offset the twin axis below the host
                self.axes2[i].spines['bottom'].set_position(('axes', -0.1*(i + 1)))
            # Else if new axes are y
            else:
                self.axes2[i] = self.axes.twinx()
                self.axes2[i].spines['right'].set_position(('axes', 1 + 0.1*i))

            # Turn on the frame for the twin axis, but then hide all
            # but the bottom/right spine
            self.axes2[i].set_frame_on(True)
            self.axes2[i].patch.set_visible(False)
            for sp in self.axes2[i].spines.values():
                sp.set_visible(False)

            if self.ax2loc == 'x':
                self.axes2[i].spines["bottom"].set_visible(True)
            else:
                self.axes2[i].spines["right"].set_visible(True)

            if self.ax2loc == 'x':

                self.axes2[i].set_xlabel(self.ax2label[i])

                self.axes2[i].tick_params(axis='x', colors=color)
                self.axes2[i].xaxis.label.set_color(color)
            else:

                self.axes2[i].set_ylabel(self.ax2label[i])
                self.axes2[i].tick_params(axis='y', colors=color)
                self.axes2[i].yaxis.label.set_color(color)

            self.axes2[i].set_xscale(xyscale2[0]), self.axes2[i].set_yscale(xyscale2[1])
            if self.x2lim is not None: self.axes2[i].set_xlim(self.x2lim)
            if self.y2lim is not None: self.axes2[i].set_ylim(self.y2lim)

            # Plot
            # Set the 2nd axis plots layer in the back. The higher zorder, the more front the plot is
            self.axes.set_zorder(self.axes2[i].get_zorder() + 1)
            if self.plot_type2[i] == 'shade':
                self.plots2[i] = self.axes2[i].fill_between(self.list_x2[i], 0, self.list_y2[i], alpha=1, facecolor=(160/255., 160/255., 160/255.),
                                                     interpolate=False)
            elif self.plot_type2[i] == 'line':
                # If show markers in line plots
                if not self.showmarker:
                    self.plots2[i], = self.axes2[i].plot(self.list_x2[i], self.list_y2[i],
                                                        ls=self.lines[i + self.narr], label=str(self.linelabel2[i]),
                                                        color=self.colors[i + self.narr], alpha=self.alpha)
                # Else if don't show markers in line plot
                else:
                    self.plots2[i], = self.axes2[i].plot(self.list_x2[i], self.list_y2[i],
                                                        ls=self.lines[i + self.narr], label=str(self.linelabel2[i]),
                                                        color=self.colors[i + self.narr], alpha=self.alpha,
                                                        marker=self.markers[i + self.narr])
                    
                    
    def finalizeFigure(self, **kwargs):
        super(Plot2D_MultiAxes, self).finalizeFigure(tight_layout=True, **kwargs)


# TODO: verify
class BaseFigure3D(BaseFigure):
    def __init__(self, listX2D, listY2D, zlabel='$z$', alpha=1, viewangle=(15, -115), zLim=(None,), cmap='plasma', cmapLabel='$U$', grid=True, cbar_orient='horizontal', **kwargs):
        super(BaseFigure3D, self).__init__(list_x=listX2D, list_y=listY2D, **kwargs)
        # The name of list_x and list_y becomes listX2D and listY2D since they are 2D
        self.listX2D, self.listY2D = self.list_x, self.list_y
        self.zlabel, self.zLim = zlabel, zLim
        self.cmapLabel, self.cmap = cmapLabel, cmap
        self.alpha, self.grid, self.viewangle = alpha, grid, viewangle
        self.plot, self.cbar_orient = None, cbar_orient


    def initializeFigure(self, figSize=(1, 1)):
        # Update Matplotlib rcparams
        self.latexify(fontsize = self.fontsize, figwidth = self.figwidth, subplots=figSize)
        self.fig = plt.figure(self.name)
        self.axes = (self.fig.gca(projection = '3d'),)
        # self.axes = (self.fig.add_subplot(111, projection='3d'),)


    def plotFigure(self):
        super(BaseFigure3D, self).plotFigure()

        self._ensureMeshGrid()


    def finalizeFigure(self, fraction = 0.06, pad = 0.08, showCbar = True, reduceNtick = True,
                       **kwargs):
        self.axes[0].set_zlabel(self.zlabel)
        self.axes[0].set_zlim(self.zLim)
        # Color bar
        if showCbar:
            cb = plt.colorbar(self.plot, fraction = fraction, pad = pad, orientation = self.cbar_orient, extend = 'both', aspect = 25, shrink = 0.75)
            cb.set_label(self.cmapLabel)

        # Turn off background on all three panes
        self._format3D_Axes(self.axes[0])
        # Equal axes
        # [REQUIRES SOURCE CODE MODIFICATION] Equal axis
        # Edit the get_proj function inside site-packages\mpl_toolkits\mplot3d\axes3d.py:
        # try: self.localPbAspect=self.pbaspect
        # except AttributeError: self.localPbAspect=[1,1,1]
        # xmin, xmax = np.divide(self.get_xlim3d(), self.localPbAspect[0])
        # ymin, ymax = np.divide(self.get_ylim3d(), self.localPbAspect[1])
        # zmin, zmax = np.divide(self.get_zlim3d(), self.localPbAspect[2])
        if self.equalaxis:
            try:
                arZX = abs((self.zLim[1] - self.zLim[0])/(self.xlim[1] - self.xlim[0]))
                arYX = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))

                # Constrain AR from getting too large
                arYX, arZX = np.min((arYX, 2)), np.min((arZX, 2))
                # Axes aspect ratio doesn't really work properly
                self.axes[0].pbaspect = (1, arYX, arZX)
                # auto_scale_xyz is not preferable since it does it by setting a cubic box
                # scaling = np.array([getattr(self.axes[0], 'get_{}lim'.format(dim))() for dim in 'xyz'])
                # self.axes[0].auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
            except AttributeError:
                warn('\nTo set custom aspect ratio of the 3D plot, you need modification of the source code axes3d.py. The aspect ratio might be incorrect for ' + self.name + '\n', stacklevel = 2)
                pass

        if reduceNtick:
            self.axes[0].set_xticks(np.linspace(self.xlim[0], self.xlim[1], 3))
            self.axes[0].set_yticks(np.linspace(self.ylim[0], self.ylim[1], 3))
            self.axes[0].set_zticks(np.linspace(self.zLim[0], self.zLim[1], 3))

        # # Strictly equal axis of all three axis
        # _, _, _, _, _, _ = self.get3D_AxesLimits(self.axes[0])
        # 3D grid
        self.axes[0].grid(self.grid)
        self.axes[0].view_init(self.viewangle[0], self.viewangle[1])
        # # View distance
        # self.axes[0].dist = 11

        super().finalizeFigure(grid = False, showleg = False, **kwargs)
        

    @timer
    @jit(parallel = True, fastmath = True)
    def getSlicesLimits(self, listX2D, listY2D, listZ2D = np.empty(100), listOtherVals = np.empty(100)):
        getXlim = True if self.xlim[0] is None else False
        getYlim = True if self.ylim[0] is None else False
        getZlim = True if self.zLim[0] is None else False
        self.xlim = [1e20, -1e20] if self.xlim[0] is None else self.xlim
        self.ylim = [1e20, -1e20] if self.ylim[0] is None else self.ylim
        self.zLim = [1e20, -1e20] if self.zLim[0] is None else self.zLim
        otherValsLim = [1e20, -1e20]
        for i in prange(len(listX2D)):
            if getXlim:
                xmin, xmax = np.min(listX2D[i]), np.max(listX2D[i])
                # Replace old limits with new ones if better limits found
                self.xlim[0] = xmin if xmin < self.xlim[0] else self.xlim[0]
                self.xlim[1] = xmax if xmax > self.xlim[1] else self.xlim[1]

            if getYlim:
                ymin, ymax = np.min(listY2D[i]), np.max(listY2D[i])
                self.ylim[0] = ymin if ymin < self.ylim[0] else self.ylim[0]
                self.ylim[1] = ymax if ymax > self.ylim[1] else self.ylim[1]

            if getZlim:
                zmin, zmax = np.min(listZ2D[i]), np.max(listZ2D[i])
                self.zLim[0] = zmin if zmin < self.zLim[0] else self.zLim[0]
                self.zLim[1] = zmax if zmax > self.zLim[1] else self.zLim[1]

            otherVals_min, otherVals_max = np.nanmin(listOtherVals[i]), np.nanmax(listOtherVals[i])
            otherValsLim[0] = otherVals_min if otherVals_min < otherValsLim[0] else otherValsLim[0]
            otherValsLim[1] = otherVals_max if otherVals_max > otherValsLim[1] else otherValsLim[1]

        return otherValsLim
         

    @staticmethod
    def _format3D_Axes(ax):
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
    def __init__(self, contourX2D, contourY2D, listSlices2D, sliceOffsets, zDir='val', contour_lvl=10, grad_bg=True, equalaxis=False, **kwargs):
        super(PlotContourSlices3D, self).__init__(listX2D=contourX2D, listY2D=contourY2D, equalaxis=equalaxis, **kwargs)

        self.listSlices2D = (listSlices2D,) if isinstance(listSlices2D, np.ndarray) else listSlices2D
        self.sliceOffsets, self.zDir = iter(sliceOffsets), zDir
        self.xlim = (min(sliceOffsets), max(sliceOffsets)) if (self.xlim[0] is None) and (zDir == 'x') else self.xlim
        self.ylim = (min(sliceOffsets), max(sliceOffsets)) if (self.ylim[0] is None) and (zDir == 'y') else self.ylim
        self.zLim = (min(sliceOffsets), max(sliceOffsets)) if (self.zLim[0] is None) and (zDir == 'val') else self.zLim
        # If axis limits are still not set, infer
        # if self.xlim[0] is None:
        #     self.xlim = (np.min(contourX2D), np.max(contourX2D))
        #
        # if self.ylim[0] is None:
        #     self.ylim = (np.min(contourY2D), np.max(contourY2D))
        #
        # if self.zLim[0] is None:
        #     self.zLim = (np.min(listSlices2D), np.max(listSlices2D))
        _ = self.getSlicesLimits(listX2D=self.listX2D, listY2D=self.listY2D, listZ2D=self.listSlices2D)
        # self.sliceMin, self.sliceMax = self.zLim
        self.sliceMin, self.sliceMax = np.amin(listSlices2D), np.amax(listSlices2D)
        self.contour_lvl, self.grad_bg = contour_lvl, grad_bg
        # # Good initial view angle
        # self.viewangle = (20, -115) if zDir is 'val' else (15, -60)
        # self.cbar_orient = 'vertical' if zDir is 'val' else 'horizontal'


    # def initializeFigure(self, figSize = (1, 1)):
    #     # If zDir is 'val', then the figure height is twice width, else, figure width is twice height
    #     # figSize = (2.75, 1) if self.zDir is 'val' else (1, 2)
    #
    #     super().initializeFigure(figSize = figSize)


    def plotFigure(self):
        super(PlotContourSlices3D, self).plotFigure()

        # Currently, gradient background feature is only available for zDir = 'x'
        if self.grad_bg:
            if self.zDir is 'x':
                x2Dbg, y2Dbg = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.linspace(self.ylim[0], self.ylim[1], 3))
                z2Dbg, _ = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.linspace(self.zLim[0], self.zLim[1], 3))
                self.axes[0].contourf(x2Dbg, y2Dbg, z2Dbg, 500, zdir = 'val', offset = 0, cmap = 'gray', alpha = 0.5, antialiased = True)
                # # Uncomment below to enable gradient background of all three planes
                # self.axes[0].contourf(x2Dbg, z2Dbg, y2Dbg, 500, zdir = 'y', offset = 300, cmap = 'gray', alpha = 0.5, antialiased = True)
                # Y3, Z3 = np.meshgrid(np.linspace(self.ylim[0], self.ylim[1], 3),
                #                      np.linspace(self.zLim[0], self.zLim[1], 3))
                # X3 = np.ones(Y3.shape)*self.xlim[0]
                # self.axes[0].plot_surface(X3, Y3, Z3, color = 'gray', alpha = 0.5)
            else:
                warn('\nGradient background only supports zDir = "x"!\n', stacklevel = 2)

        # Actual slice plots
        for i, slice in enumerate(self.listSlices2D):
            if self.zDir is 'x':
                X, Y, Z = slice, self.listX2D[i], self.listY2D[i]
            elif self.zDir is 'y':
                X, Y, Z = self.listX2D[i], slice, self.listY2D[i]
            else:
                X, Y, Z = self.listX2D[i], self.listY2D[i], slice

            # "levels" makes sure all slices are in same cmap range
            self.plot = self.axes[0].contourf(X, Y, Z, self.contour_lvl, zdir = self.zDir,
                                              offset = next(self.sliceOffsets), alpha = self.alpha, cmap = self.cmap,
                                              levels = np.linspace(self.sliceMin, self.sliceMax, 100), antialiased = False)


    def finalizeFigure(self, **kwargs):
        # Custom color bar location in the figure
        (fraction, pad) = (0.046, 0.04) if self.zDir is 'val' else (0.06, 0.08)
        # if self.zDir is 'val':
        #     ar = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))
        #     pbaspect = (1, ar, 1)
        # elif self.zDir is 'x':
        #     ar = abs((self.zLim[1] - self.zLim[0])/(self.ylim[1] - self.ylim[0]))
        #     pbaspect = (1, 1, ar)
        # else:
        #     ar = abs((self.zLim[1] - self.zLim[0])/(self.xlim[1] - self.xlim[0]))
        #     pbaspect = (1, 1, ar)

        super(PlotContourSlices3D, self).finalizeFigure(fraction = fraction, pad = pad, **kwargs)


class PlotSurfaceSlices3D(BaseFigure3D):
    def __init__(self, listX2D, listY2D, listZ2D, listSlices2D, **kwargs):
        super(PlotSurfaceSlices3D, self).__init__(listX2D = listX2D, listY2D = listY2D, **kwargs)

        self.listZ2D = (listZ2D,) if isinstance(listZ2D, np.ndarray) else listZ2D
        self.listSlices2D = (listSlices2D,) if isinstance(listSlices2D, np.ndarray) else listSlices2D
        # self.xlim = (np.min(listX2D), np.max(listX2D)) if self.xlim[0] is None else self.xlim
        # self.ylim = (np.min(listY2D), np.max(listY2D)) if self.ylim[0] is None else self.ylim
        # self.zLim = (np.min(listZ2D), np.max(listZ2D)) if self.zLim[0] is None else self.zLim
        self.cmapLim = self.getSlicesLimits(listX2D = self.listX2D, listY2D = self.listY2D, listZ2D = listZ2D,
                                           listOtherVals = listSlices2D)
        # self.listX2D, self.listY2D = iter(self.listX2D), iter(self.listY2D)

        # Find minimum and maximum of the slices values for color, ignore NaN
        # self.cmapLim = (np.nanmin(listSlices2D), np.nanmax(listSlices2D))
        self.cmapNorm = mpl.colors.Normalize(self.cmapLim[0], self.cmapLim[1])
        self.cmapVals = plt.cm.ScalarMappable(norm = self.cmapNorm, cmap = self.cmap)
        self.cmapVals.set_array([])
        # For colorbar mappable
        self.plot = self.cmapVals


    def plotFigure(self):
        for i, slice in enumerate(self.listSlices2D):
            print('\nPlotting ' + self.name + '...')
            fColors = self.cmapVals.to_rgba(slice)
            self.axes[0].plot_surface(self.listX2D[i], self.listY2D[i], self.listZ2D[i], cstride = 1,
                                      rstride = 1, facecolors = fColors, vmin = self.cmapLim[0], vmax = self.cmapLim[1], shade = False)


    # def finalizeFigure(self, **kwargs):
    #     arZX = abs((self.zLim[1] - self.zLim[0])/(self.xlim[1] - self.xlim[0]))
    #     arYX = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))
    #     # Axes aspect ratio doesn't really work properly
    #     pbaspect = (1., arYX, arZX*2)
    #
    #     super(PlotSurfaceSlices3D, self).finalizeFigure(pbaspect = pbaspect, **kwargs)


class PlotImageSlices3D(BaseFigure3D):
    def __init__(self, listX2D, listY2D, listZ2D, listRGB, **kwargs):
        super(PlotImageSlices3D, self).__init__(listX2D = listX2D, listY2D = listY2D, **kwargs)

        # Convert listZ2D to tuple if it's np.ndarray
        self.listZ2D = (listZ2D,) if isinstance(listZ2D, np.ndarray) else listZ2D
        self.listRGB = (listRGB,) if isinstance(listRGB, np.ndarray) else listRGB
        # Make sure list of RGB arrays are between 0 and 1
        for i, rgbVals in enumerate(self.listRGB):
            self.listRGB[i][rgbVals > 1] = 1
            self.listRGB[i][rgbVals < 0] = 0
                
        # Axes limits
        # self.xlim = (np.min(listX2D), np.max(listX2D)) if self.xlim[0] is None else self.xlim
        # self.ylim = (np.min(listY2D), np.max(listY2D)) if self.ylim[0] is None else self.ylim
        # self.zLim = (np.min(listZ2D), np.max(listZ2D)) if self.zLim[0] is None else self.zLim
        _ = self.getSlicesLimits(listX2D = self.listX2D, listY2D = self.listY2D, listZ2D = self.listZ2D)



    @timer
    @jit(parallel = True)
    def plotFigure(self):
        print('\nPlotting {}...'.format(self.name))
        # For gauging progress
        milestone = 33
        for i in prange(len(self.listRGB)):
            self.axes[0].plot_surface(self.listX2D[i], self.listY2D[i], self.listZ2D[i], cstride = 1, rstride = 1, 
                                      facecolors = self.listRGB[i], shade = False)
            progress = (i + 1)/len(self.listRGB)*100.
            if progress >= milestone:
                print(' {0}%... '.format(milestone))
                milestone += 33
            
    
    def finalizeFigure(self, **kwargs):
        super(PlotImageSlices3D, self).finalizeFigure(showCbar = False, **kwargs)
            
    
    
        
        

        











if __name__ == '__main__':
    x = np.linspace(0, 300, 100)
    y = np.linspace(0, 100, 100)
    y2 = np.linspace(10, 80, 100)

    val = np.linspace(1, 10, x.size*y.size).reshape((y.size, x.size))
    z2D2 = np.linspace(10, 30, x.size*y.size).reshape((y.size, x.size))
    z2D3 = np.linspace(30, 60, x.size*y.size).reshape((y.size, x.size))

    # myplot = Plot2D_InsetZoom((x, x), (y, y2), val = (None,), zoomBox = (10, 60, 20, 40), save = True, equalaxis = True, figdir = 'R:/', name = 'newFig')

    # myplot = Plot2D_InsetZoom(x, y, val = val, zoomBox = (10, 70, 10, 30), save = True, equalaxis = True,
    #                           figdir = 'R:/', name = 'newFig2')

    # myplot = PlotSlices3D(x, y, [val, z2D2, z2D3], sliceOffsets = [0, 20, 50], name = '3d2', figdir = 'R:/', xlim = (0, 150), zDir = 'x')
    myplot = PlotContourSlices3D(x, y, [val, z2D2, z2D3], sliceOffsets = [20000, 20500, 21000], name = '3d2', figdir = 'R:/', zDir = 'x', xlabel = '$x$', ylabel = '$y$', zlabel = r'$z$ [m]', zLim = (0, 100), ylim = (0, 300), grad_bg = True)

    myplot.initializeFigure()

    myplot.plotFigure()

    myplot.finalizeFigure()


