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
from scipy import ndimage
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from copy import copy
import matplotlib.cm

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


    def _latexify(self, fig_width=None, fig_height=None, figspan='half', linewidth=0.8, fontsize=8, subplots=(1, 1), figheight_multiplier=1.,
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
            warn('\nx and y are 1D, contour/contourf requires mesh grid. Converting x and y to mesh grid '
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
        self.lines, self.markers = ("-", "--", "-.", ":")*5, ('o', 'D', 's', 'v', '^', '<', '>', 's', '8', 'p')*3
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


    def plotFigure(self, linelabel=None, showmarker=False, contour_lvl=20, markercolors=None, **kwargs):
        self.markercolors = (markercolors,)*self.narr if isinstance(markercolors, np.ndarray) else markercolors
        self.showmarker, self.contour_lvl = showmarker, contour_lvl
        # Gradient background, only for line and scatter plots
        if self.grad_bg and self.plot_type[0] in ('line', 'scatter'):
            xmesh, ymesh = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.linspace(self.ylim[0], self.ylim[1], 3))
            graymesh = (np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.arange(3)))[0] if self.grad_bg_dir is 'x' else (np.meshgrid(np.arange(3), np.linspace(self.ylim[0], self.ylim[1], 3)))[1]
            self.axes.contourf(xmesh, ymesh, graymesh, 500, cmap='gray', alpha=0.333, vmin=self.grad_bg_range[0], vmax=self.grad_bg_range[1])

        super().plotFigure(**kwargs)

        self.linelabel = np.arange(1, len(self.list_x) + 1) if linelabel is None else linelabel
        self.plots = [None]*self.narr
        for i in range(self.narr):
            if self.plot_type[i] == 'line':
                if not showmarker:
                    self.plots[i] = self.axes.plot(self.list_x[i], self.list_y[i], ls=self.lines[i],
                                                   label=str(self.linelabel[i]), color=self.colors[i],
                                                   alpha=self.alpha)

                else:
                    self.plots[i] = self.axes.plot(self.list_x[i], self.list_y[i], ls=self.lines[i],
                                                   label=str(self.linelabel[i]), color=self.colors[i],
                                                   alpha=self.alpha,
                                                   marker=self.markers[i])
                    
            elif self.plot_type[i] == 'scatter':
                if self.markercolors is None:
                    self.plots[i] = self.axes.scatter(self.list_x[i], self.list_y[i], lw=0, s=10,
                                                      label=str(self.linelabel[i]), alpha=self.alpha, color=self.colors[i], marker=self.markers[i])
                else:
                    self.plots[i] = self.axes.scatter(self.list_x[i], self.list_y[i], lw=0, s=10,
                                                      label=str(self.linelabel[i]), alpha=self.alpha,
                                                      c=self.markercolors[i], marker=self.markers[i])

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
        if self.plot_type[0] in ('contourf', 'contour') or (showcb and self.markercolors is not None):
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
        self._mark_inset(self.axes, self.axes[1], loc1a = 1, loc1b = 4, loc2a = 2, loc2b = 3, fc = "none",
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
                self.axes.spines[spine].set_visible(False)
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
        self.ax2label = (ax2label,)*self.narr2 if isinstance(ax2label, str) else ax2label
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
                self.axes2[i].spines['bottom'].set_position(('axes', -0.3*(i + 1)))
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
                # If don't show markers in line plots
                if not self.showmarker:
                    self.plots2[i], = self.axes2[i].plot(self.list_x2[i], self.list_y2[i],
                                                        ls=self.lines[i + self.narr], label=str(self.linelabel2[i]),
                                                        color=self.colors[i + self.narr], alpha=self.alpha)
                # Else if show markers in line plot
                else:
                    self.plots2[i], = self.axes2[i].plot(self.list_x2[i], self.list_y2[i],
                                                        ls=self.lines[i + self.narr], label=str(self.linelabel2[i]),
                                                        color=self.colors[i + self.narr], alpha=self.alpha,
                                                        marker=self.markers[i + self.narr])
                    
                    
    def finalizeFigure(self, tight_layout=True, **kwargs):
        super(Plot2D_MultiAxes, self).finalizeFigure(tight_layout=tight_layout, **kwargs)


class Plot3D(BaseFigure):
    def __init__(self, list_x, list_y, list_z,
                 zlabel='$z$',
                 viewangle=(15, -115), zlim=None, cmap='plasma', val_label='Value', zdir='z', equalaxis=True, 
                 **kwargs):
        self.list_z = (list_z,) if isinstance(list_z, np.ndarray) else list_z
        self.zlabel=zlabel
        self.viewangle = viewangle
        self.zlim=zlim
        self.cmap = cmap
        self.val_label = val_label
        self.zdir = zdir
        super(Plot3D, self).__init__(list_x, list_y, equalaxis=equalaxis, **kwargs)
        _ = self._getSlicesLimits(list_x, list_y, list_z)

    def initializeFigure(self, **kwargs):
        self._latexify(**kwargs)
        self.fig = plt.figure(self.name, constrained_layout=True)
        self.axes = self.fig.add_subplot(111, projection='3d')

    def plotFigure(self, **kwargs):
        for i in range(self.narr):
            self.axes.plot(self.list_x[i], self.list_y[i], self.list_z[i], zdir=self.zdir, label=self.zlabel)
            
    def finalizeFigure(self, grid=True, **kwargs):
        self.axes.set_zlabel(self.zlabel)
        if self.zlim is not None: self.axes.set_zlim(self.zlim)
        self._format3D_Axes(self.axes)
        if self.equalaxis:
            try:
                ar_zx = abs((self.zlim[1] - self.zlim[0])/(self.xlim[1] - self.xlim[0]))
                ar_yx = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))
                # Constrain AR from getting too large
                ar_yx, ar_zx = np.min((ar_yx, 2)), np.min((ar_zx, 2))
                # Axes aspect ratio doesn't really work properly
                self.axes.pbaspect = (1, ar_yx, ar_zx)
                # auto_scale_xyz is not preferable since it does it by setting a cubic box
                # scaling = np.array([getattr(self.axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                # self.axes.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
            except AttributeError:
                warn('\nTo set custom aspect ratio of the 3D plot, you need modification of the source code axes3d.py. The aspect ratio might be incorrect for ' + self.name + '\n', stacklevel=2)

        self.axes.grid(grid)
        self.axes.view_init(self.viewangle[0], self.viewangle[1])
        super(Plot3D, self).finalizeFigure(grid=False, showleg=False, **kwargs)

    @staticmethod
    def _format3D_Axes(ax):
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_yaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_zaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})

    def _getSlicesLimits(self, list_x, list_y, list_z=np.empty(100), list_val=np.empty(100)):
        get_xlim = True if self.xlim is None else False
        get_ylim = True if self.ylim is None else False
        get_zlim = True if self.zlim is None else False
        self.xlim = [1e20, -1e20] if self.xlim is None else self.xlim
        self.ylim = [1e20, -1e20] if self.ylim is None else self.ylim
        self.zlim = [1e20, -1e20] if self.zlim is None else self.zlim
        val_lim = [1e20, -1e20]
        for i in range(len(list_x)):
            if get_xlim:
                xmin, xmax = np.min(list_x[i].ravel()), np.max(list_x[i].ravel())
                # Replace old limits with new ones if better limits found
                self.xlim[0] = xmin if xmin < self.xlim[0] else self.xlim[0]
                self.xlim[1] = xmax if xmax > self.xlim[1] else self.xlim[1]

            if get_ylim:
                ymin, ymax = np.min(list_y[i].ravel()), np.max(list_y[i].ravel())
                self.ylim[0] = ymin if ymin < self.ylim[0] else self.ylim[0]
                self.ylim[1] = ymax if ymax > self.ylim[1] else self.ylim[1]

            if get_zlim:
                zmin, zmax = np.min(list_z[i]), np.max(list_z[i])
                self.zlim[0] = zmin if zmin < self.zlim[0] else self.zlim[0]
                self.zlim[1] = zmax if zmax > self.zlim[1] else self.zlim[1]

            val_min, val_max = np.nanmin(list_val[i].ravel()), np.nanmax(list_val[i].ravel())
            val_lim[0] = val_min if val_min < val_lim[0] else val_lim[0]
            val_lim[1] = val_max if val_max > val_lim[1] else val_lim[1]

        return val_lim


# FIXME: not working
class Plot3D_WindFarmSlices(Plot3D):
    def __init__(self, list_x1, list_x2, list_val, case, offsets, turblocs=(0., 0.),
                 val_lim=None, offset_lim=None,
                 slicenames=None, flowangle=30, turbr=63., turb_pitch=5,
                 equalaxis=True, **kwargs):
        super(Plot3D_WindFarmSlices, self).__init__(list_x=list_x1, list_y=list_x2, list_z=offsets, zlim=offset_lim, equalaxis=equalaxis, **kwargs)
        self.list_val = (list_val,) if isinstance(list_val, np.ndarray) else list_val
        self.slicenames = (slicenames,) if isinstance(slicenames, str) else slicenames
        self.offsets = offsets
        self.flowangle = flowangle/180.*np.pi if flowangle > 2.*np.pi else flowangle
        self.turblocs = (turblocs,)*2 if 'OneTurb' in case else turblocs
        self.turb_pitch = turb_pitch/180.*np.pi if turb_pitch > np.pi else turb_pitch
        self.case = case
        self.val_lim = (val_lim,)*2 if val_lim is None else val_lim
        if abs(self.list_y[0].ravel().min() - self.list_y[0].ravel().max()) < 1.:
            self.slice_type = 'top'
            if 'OneTurb' in case:
                self.turblocs = ((1118.083, 1279.5),)*2
            elif 'SeqTurb' in case:
                self.turblocs = ((1118.083, 1279.5), (1881.917, 1720.5))
            else:
                self.turblocs = ((1244.083, 1061.262), (992.083, 1497.738))
        else:
            self.zdir = 'z'
            if 'alongWind' in slicenames[0]:
                self.slice_type ='side'
                if 'OneTurb' in case:
                    self.turblocs = ((1291.051, 90.),)*2
                else:
                    self.turblocs = ((1436.543, 90.), (1150.082, 90.))
            else:
                self.slice_type = 'front'

        if self.slice_type == 'front':
            patch1 = Circle(self.turblocs[0], turbr, alpha=0.5, fill=False, edgecolor=(0.25,)*3)
            patch2 = Circle(self.turblocs[1], turbr, alpha=0.5, fill=False, edgecolor=(0.25,)*3)
            self.patches = []
            for _ in range(self.narr*2):
                self.patches.append(copy(patch1))
                self.patches.append(copy(patch2))

            self.patches = iter(self.patches)
        elif self.slice_type == 'side':
            self.line1 = [[self.turblocs[0][0] - turbr*np.sin(self.turb_pitch), self.turblocs[0][1] - turbr],
                     [self.turblocs[0][0] + turbr*np.sin(self.turb_pitch), self.turblocs[0][1] + turbr]]
            self.line2 = [[self.turblocs[1][0] - turbr*np.sin(self.turb_pitch), self.turblocs[1][1] - turbr],
                          [self.turblocs[1][0] + turbr*np.sin(self.turb_pitch), self.turblocs[1][1] + turbr]]
        elif self.slice_type == 'top':
            if 'SeqTurb' in case:
                self.line1 = [[1149.583, 1224.94], [1086.583, 1334.06]]
                self.line2 = [[1913.417, 1665.94], [1850.417, 1775.06]]
            elif 'OneTurb' in case:
                self.line1 = self.line2 = [[1149.583, 1224.94], [1086.583, 1334.06]]
            elif 'ParTurb' in case:
                self.line1 = [[1275.583, 1006.702], [1212.583, 1115.821]]
                self.line2 = [[1023.583, 1443.179], [960.583, 1552.298]]

    def plotFigure(self, **kwargs):
        for i in range(self.narr):
            if self.slice_type == 'front':
                patch1 = next(self.patches)
                patch2 = next(self.patches)
                self.axes.add_patch(patch1)
                art3d.pathpatch_2d_to_3d(patch1, z=self.offsets[i], zdir=self.zdir)
                if 'OneTurb' not in self.case:
                    self.axes.add_patch(patch2)
                    art3d.pathpatch_2d_to_3d(patch2, z=self.offsets[i], zdir=self.zdir)
            else:
                self.axes.plot(self.line1[0], self.line1[1], np.ones(2)*self.offsets[i], alpha=0.5, c=(0.25, 0.25, 0.25), zdir=self.zdir)
                if 'OneTurb' not in self.case:
                    self.axes.plot(self.line2[0], self.line2[1], np.ones(2)*self.offsets[i], alpha=0.5, c=(0.25, 0.25, 0.25), zdir=self.zdir)

            if self.slice_type in ('front', 'side'):
                self.axes.plot_surface(self.list_x[i], self.list_y[i], np.ones_like(self.list_x[i])*self.offsets[i], rstride=1, cstride=1, facecolors=plt.cm.plasma(self.list_val[i]), shade=False,
                                     vmin=self.val_lim[0], vmax=self.val_lim[1])
            else:
                self.axes.plot_surface(self.list_x[i], self.list_y[i], np.ones_like(self.list_x[i])*self.offsets[i],
                                     rstride=1, cstride=1, facecolors=plt.cm.plasma(self.list_val[i]), shade=False,
                                     vmin=self.val_lim[0], vmax=self.val_lim[1])
                

    
                



class BaseFigure3D(BaseFigure):
    def __init__(self, list_x, list_y, zlabel='$z$', alpha=1, viewangle=(15, -115), zlim=None, cmap='plasma', val_label='$U$', cbar_orient='horizontal', **kwargs):
        super(BaseFigure3D, self).__init__(list_x=list_x, list_y=list_y, **kwargs)
        # The name of list_x and list_y becomes list_x and list_y since they are 2D
        self.list_x, self.list_y = self.list_x, self.list_y
        self.zlabel, self.zlim = zlabel, zlim
        self.val_label, self.cmap = val_label, cmap
        self.alpha, self.viewangle = alpha, viewangle
        self.plot, self.cbar_orient = None, cbar_orient

    def initializeFigure(self, constrained_layout=True, proj_type = 'persp', **kwargs):
        # Update Matplotlib rcparams
        self._latexify(fontsize=self.fontsize, figspan=self.figwidth, figheight_multiplier=self.figheight_multiplier, **kwargs)
        self.fig = plt.figure(self.name, constrained_layout=constrained_layout)
        self.axes = self.fig.add_subplot(111, projection='3d', proj_type=proj_type)

    def plotFigure(self):
        super(BaseFigure3D, self).plotFigure()
        self._ensureMeshGrid()

    def finalizeFigure(self, fraction=0.06, pad=0.08, show_cbar=True, reduce_nticks=True, grid=True, show_zlabel=True,
                       **kwargs):
        if show_zlabel: self.axes.set_zlabel(self.zlabel)
        if self.zlim is not None: self.axes.set_zlim(self.zlim)
        # Color bar
        if show_cbar:
            cb = plt.colorbar(self.plot, fraction=fraction, pad=pad, orientation=self.cbar_orient, extend='both', aspect=25, shrink=0.75)
            cb.set_label(self.val_label)

        # Turn off background on all three panes
        self._format3D_Axes(self.axes)
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
                ar_zx = abs((self.zlim[1] - self.zlim[0])/(self.xlim[1] - self.xlim[0]))
                ar_yx = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))

                # Constrain AR from getting too large
                ar_yx, ar_zx = np.min((ar_yx, 2)), np.min((ar_zx, 2))
                # Axes aspect ratio doesn't really work properly
                self.axes.pbaspect = (1, ar_yx, ar_zx)
                # auto_scale_xyz is not preferable since it does it by setting a cubic box
                # scaling = np.array([getattr(self.axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                # self.axes.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
            except AttributeError:
                warn('\nTo set custom aspect ratio of the 3D plot, you need modification of the source code axes3d.py. The aspect ratio might be incorrect for ' + self.name + '\n', stacklevel = 2)

        if reduce_nticks:
            if self.xlim is not None: self.axes.set_xticks(np.linspace(self.xlim[0], self.xlim[1], 3))
            if self.ylim is not None: self.axes.set_yticks(np.linspace(self.ylim[0], self.ylim[1], 3))
            if self.zlim is not None: self.axes.set_zticks(np.linspace(self.zlim[0], self.zlim[1], 3))

        # # Strictly equal axis of all three axis
        # _, _, _, _, _, _ = self.get3D_AxesLimits(self.axes)
        # 3D grid
        self.axes.grid(grid)
        self.axes.view_init(self.viewangle[0], self.viewangle[1])
        # # View distance
        # self.axes.dist = 11
        super().finalizeFigure(grid=False, showleg=False, xyscale=(None, None), **kwargs)
        
    def _getSlicesLimits(self, list_x, list_y, list_z = np.empty(100), list_val = np.empty(100)):
        get_xlim = True if self.xlim is None else False
        get_ylim = True if self.ylim is None else False
        get_zlim = True if self.zlim is None else False
        self.xlim = [1e20, -1e20] if self.xlim is None else self.xlim
        self.ylim = [1e20, -1e20] if self.ylim is None else self.ylim
        self.zlim = [1e20, -1e20] if self.zlim is None else self.zlim
        val_lim = [1e20, -1e20]
        for i in range(len(list_x)):
            if get_xlim:
                xmin, xmax = np.min(list_x[i].ravel()), np.max(list_x[i].ravel())
                # Replace old limits with new ones if better limits found
                self.xlim[0] = xmin if xmin < self.xlim[0] else self.xlim[0]
                self.xlim[1] = xmax if xmax > self.xlim[1] else self.xlim[1]

            if get_ylim:
                ymin, ymax = np.min(list_y[i].ravel()), np.max(list_y[i].ravel())
                self.ylim[0] = ymin if ymin < self.ylim[0] else self.ylim[0]
                self.ylim[1] = ymax if ymax > self.ylim[1] else self.ylim[1]

            if get_zlim:
                zmin, zmax = np.min(list_z[i].ravel()), np.max(list_z[i].ravel())
                self.zlim[0] = zmin if zmin < self.zlim[0] else self.zlim[0]
                self.zlim[1] = zmax if zmax > self.zlim[1] else self.zlim[1]

            val_min, val_max = np.nanmin(list_val[i].ravel()), np.nanmax(list_val[i].ravel())
            val_lim[0] = val_min if val_min < val_lim[0] else val_lim[0]
            val_lim[1] = val_max if val_max > val_lim[1] else val_lim[1]

        return val_lim
         
    @staticmethod
    def _format3D_Axes(ax):
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_yaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})
        ax.w_zaxis._axinfo['grid'].update({'linewidth': 0.25, 'color': 'gray'})

    @staticmethod
    def _get3D_AxesLimits(ax, set_axesequal=True):
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
        if set_axesequal:
            plot_radius = 0.5*max([x_range, y_range, z_range])
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([0, z_middle + plot_radius])

        return x_range, y_range, z_range, x_limits, y_limits, z_limits


class PlotContourSlices3D(BaseFigure3D):
    def __init__(self, list_x, list_y, list_val, slice_offsets, 
                 val_lim=None, zdir='val', grad_bg=False, equalaxis=False, **kwargs):
        super(PlotContourSlices3D, self).__init__(list_x=list_x, list_y=list_y, equalaxis=equalaxis, **kwargs)
        self.list_val = (list_val,) if isinstance(list_val, np.ndarray) else list_val
        self.slice_offsets, self.zdir = iter(slice_offsets), zdir
        self.xlim = (min(slice_offsets), max(slice_offsets)) if (self.xlim is None) and (zdir == 'x') else self.xlim
        self.ylim = (min(slice_offsets), max(slice_offsets)) if (self.ylim is None) and (zdir == 'y') else self.ylim
        self.zlim = (min(slice_offsets), max(slice_offsets)) if (self.zlim is None) and (zdir == 'val') else self.zlim
        # If axis limits are still not set, infer
        # if self.xlim[0] is None:
        #     self.xlim = (np.min(list_x), np.max(list_x))
        #
        # if self.ylim[0] is None:
        #     self.ylim = (np.min(list_y), np.max(list_y))
        #
        # if self.zlim[0] is None:
        #     self.zlim = (np.min(list_val), np.max(list_val))
        # _ = self._getSlicesLimits(list_x=self.list_x, list_y=self.list_y, list_z=self.list_val)
        # self.slicemin, self.slicemax = self.zlim
        self.slicemin, self.slicemax = np.amin(list_val), np.amax(list_val)
        self.val_lim = (self.slicemin, self.slicemax) if val_lim is None else val_lim
        self.grad_bg = grad_bg
        # # Good initial view angle
        # self.viewangle = (20, -115) if zdir is 'val' else (15, -60)
        # self.cbar_orient = 'vertical' if zdir is 'val' else 'horizontal'

    # def initializeFigure(self, figSize = (1, 1)):
    #     # If zdir is 'val', then the figure height is twice width, else, figure width is twice height
    #     # figSize = (2.75, 1) if self.zdir is 'val' else (1, 2)
    #
    #     super().initializeFigure(figSize = figSize)

    def plotFigure(self, contour_lvl=20, **kwargs):
        super(PlotContourSlices3D, self).plotFigure()
        # Currently, gradient background feature is only available for zdir = 'x'
        if self.grad_bg:
            if self.zdir is 'x':
                xbg, ybg = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.linspace(self.ylim[0], self.ylim[1], 3))
                zbg, _ = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], 3), np.linspace(self.zlim[0], self.zlim[1], 3))
                self.axes.contourf(xbg, ybg, zbg, 500, zdir='val', offset=0, cmap='gray', alpha=0.5, antialiased=True)
                # # Uncomment below to enable gradient background of all three planes
                # self.axes.contourf(xbg, zbg, ybg, 500, zdir = 'y', offset = 300, cmap = 'gray', alpha = 0.5, antialiased = True)
                # Y3, Z3 = np.meshgrid(np.linspace(self.ylim[0], self.ylim[1], 3),
                #                      np.linspace(self.zlim[0], self.zlim[1], 3))
                # X3 = np.ones(Y3.shape)*self.xlim[0]
                # self.axes.plot_surface(X3, Y3, Z3, color = 'gray', alpha = 0.5)
            else:
                warn('\nGradient background only supports zdir = "x"!\n', stacklevel = 2)

        # Actual slice plots
        for i, slice in enumerate(self.list_val):
            if self.zdir is 'x':
                x, y, z = slice, self.list_x[i], self.list_y[i]
            elif self.zdir is 'y':
                x, y, z = self.list_x[i], slice, self.list_y[i]
            else:
                x, y, z = self.list_x[i], self.list_y[i], slice

            # # "levels" makes sure all slices are in same cmap range
            # self.plot = self.axes.contourf(x, y, z, zdir=self.zdir,
            #                                offset=next(self.slice_offsets), alpha=self.alpha, cmap=self.cmap, levels=np.linspace(self.val_lim[0], self.val_lim[1], contour_lvl))
            self.plot = self.axes.plot_surface(np.ones_like(y)*(-i), y, z, rstride=1, cstride=1, facecolors=plt.cm.plasma(x), shade=False)

    def finalizeFigure(self, **kwargs):
        # Custom color bar location in the figure
        (fraction, pad) = (0.046, 0.04) if self.zdir is 'val' else (0.06, 0.08)
        # if self.zdir is 'val':
        #     ar = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))
        #     pbaspect = (1, ar, 1)
        # elif self.zdir is 'x':
        #     ar = abs((self.zlim[1] - self.zlim[0])/(self.ylim[1] - self.ylim[0]))
        #     pbaspect = (1, 1, ar)
        # else:
        #     ar = abs((self.zlim[1] - self.zlim[0])/(self.xlim[1] - self.xlim[0]))
        #     pbaspect = (1, 1, ar)
        super(PlotContourSlices3D, self).finalizeFigure(fraction=fraction, pad=pad, **kwargs)


class PlotSurfaceSlices3D(BaseFigure3D):
    def __init__(self, list_x, list_y, list_z, list_val, val_lim=None, **kwargs):
        super(PlotSurfaceSlices3D, self).__init__(list_x=list_x, list_y=list_y, **kwargs)
        self.list_z = (list_z,) if isinstance(list_z, np.ndarray) else list_z
        self.list_val = (list_val,) if isinstance(list_val, np.ndarray) else list_val
        # self.xlim = (np.min(list_x), np.max(list_x)) if self.xlim[0] is None else self.xlim
        # self.ylim = (np.min(list_y), np.max(list_y)) if self.ylim[0] is None else self.ylim
        # self.zlim = (np.min(list_z), np.max(list_z)) if self.zlim[0] is None else self.zlim
        self.cmaplim = self._getSlicesLimits(list_x=self.list_x, list_y=self.list_y, list_z=self.list_z,
                                           list_val=self.list_val)
        self.val_lim = self.cmaplim if val_lim is None else val_lim
        # Find minimum and maximum of the slices values for color, ignore NaN
        # self.cmaplim = (np.nanmin(list_val), np.nanmax(list_val))
        self.cmapnorm = mpl.colors.Normalize(self.val_lim[0], self.val_lim[1])
        self.cmapval = plt.cm.ScalarMappable(norm=self.cmapnorm, cmap=self.cmap)
        self.cmapval.set_array([])
        # For colorbar mappable
        self.plot = self.cmapval

    @timer
    def plotFigure(self):
        for i, slice in enumerate(self.list_val):
            print('\nPlotting ' + self.name + ' slice ' + str(i) + '...')
            fcolor = self.cmapval.to_rgba(slice)
            self.axes.plot_surface(self.list_x[i], self.list_y[i], self.list_z[i], cstride=1,
                                      rstride=1, facecolors=fcolor, vmin=self.val_lim[0], vmax=self.val_lim[1], shade=False)

    # def finalizeFigure(self, **kwargs):
    #     ar_zx = abs((self.zlim[1] - self.zlim[0])/(self.xlim[1] - self.xlim[0]))
    #     ar_yx = abs((self.ylim[1] - self.ylim[0])/(self.xlim[1] - self.xlim[0]))
    #     # Axes aspect ratio doesn't really work properly
    #     pbaspect = (1., ar_yx, ar_zx*2)
    #
    #     super(PlotSurfaceSlices3D, self).finalizeFigure(pbaspect = pbaspect, **kwargs)


class PlotImageSlices3D(BaseFigure3D):
    def __init__(self, list_x, list_y, list_z, list_rgb, **kwargs):
        super(PlotImageSlices3D, self).__init__(list_x=list_x, list_y=list_y, **kwargs)

        # Convert list_z to tuple if it's np.ndarray
        self.list_z = (list_z,) if isinstance(list_z, np.ndarray) else list_z
        self.list_rgb = (list_rgb,) if isinstance(list_rgb, np.ndarray) else list_rgb
        # Make sure list of RGB arrays are between 0 and 1
        for i, rgbval in enumerate(self.list_rgb):
            self.list_rgb[i][rgbval > 1.] = 1.
            self.list_rgb[i][rgbval < 0.] = 0.
                
        # Axes limits
        # self.xlim = (np.min(list_x), np.max(list_x)) if self.xlim[0] is None else self.xlim
        # self.ylim = (np.min(list_y), np.max(list_y)) if self.ylim[0] is None else self.ylim
        # self.zlim = (np.min(list_z), np.max(list_z)) if self.zlim[0] is None else self.zlim
        _ = self._getSlicesLimits(list_x=self.list_x, list_y=self.list_y, list_z=self.list_z)

    def initializeFigure(self, constrained_layout=False, **kwargs):
        super(PlotImageSlices3D, self).initializeFigure(constrained_layout=constrained_layout, **kwargs)

    @timer
    def plotFigure(self):
        print('\nPlotting {}...'.format(self.name))
        # For gauging progress
        milestone = 33
        for i in range(len(self.list_rgb)):
            self.axes.plot_surface(self.list_x[i], self.list_y[i], self.list_z[i], cstride=1, rstride=1, 
                                      facecolors=self.list_rgb[i], shade=False)
            progress = (i + 1)/len(self.list_rgb)*100.
            if progress >= milestone:
                print(' {0}%... '.format(milestone))
                milestone += 33
            
    def finalizeFigure(self, tight_layout=True, **kwargs):
        super(PlotImageSlices3D, self).finalizeFigure(show_cbar=False, tight_layout=tight_layout, **kwargs)
            
    
    
        
        

        











if __name__ == '__main__':
    # x = np.linspace(0, 300, 100)
    # y = np.linspace(0, 100, 100)
    # y2 = np.linspace(10, 80, 100)
    # val = np.linspace(1, 10, x.size*y.size).reshape((y.size, x.size))
    # z2D2 = np.linspace(10, 30, x.size*y.size).reshape((y.size, x.size))
    # z2D3 = np.linspace(30, 60, x.size*y.size).reshape((y.size, x.size))
    #
    # # myplot = Plot2D_InsetZoom((x, x), (y, y2), val = (None,), zoomBox = (10, 60, 20, 40), save = True, equalaxis = True, figdir = 'R:/', name = 'newFig')
    #
    # # myplot = Plot2D_InsetZoom(x, y, val = val, zoomBox = (10, 70, 10, 30), save = True, equalaxis = True,
    # #                           figdir = 'R:/', name = 'newFig2')
    #
    # # myplot = PlotSlices3D(x, y, [val, z2D2, z2D3], slice_offsets = [0, 20, 50], name = '3d2', figdir = 'R:/', xlim = (0, 150), zdir = 'x')
    # myplot = PlotContourSlices3D(x, y, [val, z2D2, z2D3], slice_offsets = [20000, 20500, 21000], name = '3d2', figdir = 'R:/', zdir = 'x', xlabel = '$x$', ylabel = '$y$', zlabel = r'$z$ [m]', zlim = (0, 100), ylim = (0, 300), grad_bg = True)
    #
    # myplot.initializeFigure()
    #
    # myplot.plotFigure()
    #
    # myplot.finalizeFigure()



    # # create a 21 x 21 vertex mesh
    # xx, yy = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    #
    # # create vertices for a rotated mesh (3D rotation matrix)
    # x = xx
    # y = yy
    # z = 10*np.ones(x.shape)
    #
    # # create some dummy data (20 x 20) for the image
    # data = np.cos(xx)*np.cos(xx) + np.sin(yy)*np.sin(yy)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Draw a circle on the x=0 'wall'
    # p = Circle((0.2, 0.2), 0.3, alpha=0.25, fill=False, edgecolor=(0.25, 0.25, 0.25))
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=10, zdir="z")
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.plasma(data), shade=False)
    #
    # p2 = Circle((0.4, 0.4), 0.1)
    # ax.add_patch(p2)
    # art3d.pathpatch_2d_to_3d(p2, z=5, zdir="z")
    # ax.plot_surface(x, y, z/2., rstride=1, cstride=1, facecolors=plt.cm.viridis(data), shade=False)

    c1 = np.linspace(1000, 2000, 100)
    c2 = np.linspace(1500, 1800, 100)
    c1mesh, c2mesh = np.meshgrid(c1, c2)
    val1mesh = np.random.rand(100, 100)
    val2mesh = 3.*np.random.rand(100, 100)

    # myplot = Plot3D_WindFarmSlices((c1mesh,)*2, (c2mesh,)*2, (val1mesh, val2mesh),
    #                                case='OneTurb', offsets=(15, 105),
    #                                offset_lim=(0, 200),
    #                                turblocs=(1234, 1600), val_lim=None,
    #                                slicenames=('hub', 'apex'),
    #                                show=True, save=False,
    #                                equalaxis=True)
    # myplot.initializeFigure()
    # myplot.plotFigure()
    # myplot.finalizeFigure()
    
    myplot = PlotContourSlices3D((c1mesh,)*2, (c2mesh,)*2, (val1mesh, val2mesh), (0, 10), val_lim=(0.5, 1), zdir='x', show=True, save=False)
    myplot.initializeFigure()
    myplot.plotFigure()
    myplot.finalizeFigure()







