import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numba import njit
from PlottingTool import BaseFigure3D

case = 'both'  # 'ParTurb' 'Offset'
domainSize = (3000, 3000, 1000)
origin = (0, 0, 0)

figdir = '/media/yluan'
name = 'ProbesFront'
figwidth = 'half'
xlabel = r'$x$ [m]'
ylabel = r'$y$ [m]'
zlabel = r'$z$ [m]'
legLoc, alpha = 'upper center', 0.5
markersize, scatter_depthshade = 10, True
viewangle = (15, -150)
show, save = False, True

@njit
def generateFrameCoordinates(origin=(0, 0, 0), frame_size=(3000, 3000, 1000), rotation=0):
    extent = frame_size
    # Bottom and then upper frame
    xs, ys, zs = [origin[0]], [origin[1]], [origin[2]]
    for i in range(2):
        xs += [xs[-1] + extent[0]*np.cos(rotation)]
        xs += [xs[-1] - extent[1]*np.sin(rotation)]
        xs += [xs[-1] - extent[0]*np.cos(rotation)]
        xs += [origin[0]]*2

        ys += [ys[-1] + extent[0]*np.sin(rotation)]
        ys += [ys[-1] + extent[1]*np.cos(rotation)]
        ys += [ys[-1] - extent[0]*np.sin(rotation)]
        ys += [origin[1]]*2

    xs += [xs[-1] + extent[0]*np.cos(rotation)]*2
    xs += [xs[-1] - extent[1]*np.sin(rotation)]*2
    xs += [xs[-1] - extent[0]*np.cos(rotation)]*2

    ys += [ys[-1] + extent[0]*np.sin(rotation)]*2
    ys += [ys[-1] + extent[1]*np.cos(rotation)]*2
    ys += [ys[-1] - extent[0]*np.sin(rotation)]*2

    zs += [origin[2]]*4
    zs += [extent[2]]*5
    zs += [origin[2]]*2
    zs += [extent[2]]*2
    zs += [origin[2]]*2
    zs += [extent[2]]

    return xs, ys, zs


# Domain frame
xdomain, ydomain, zdomain = generateFrameCoordinates(frame_size=domainSize, rotation=0)

if case == 'both':
    # Parallel turbine inner refinement frame
    xrefine1, yrefine1, zrefine1 = generateFrameCoordinates(origin=(1120.344, 771.583, 0), frame_size=(882, 882, 279), rotation=np.pi/6)

    # Parallel turbine outer refinement frame
    xrefine2, yrefine2, zrefine2 = generateFrameCoordinates(origin=(1074.225, 599.464, 0), frame_size=(1134, 1134, 405), rotation=np.pi/6)

    # Offset turbine inner refinement frame
    xrefine3, yrefine3, zrefine3 = generateFrameCoordinates(origin=(994.344, 989.821, 0),
                                                            frame_size=(1764, 378, 279), rotation=np.pi/6)

    # Offset turbine outer refinement frame
    xrefine4, yrefine4, zrefine4 = generateFrameCoordinates(origin=(948.225, 817.702, 0),
                                                            frame_size=(1890, 630, 405), rotation=np.pi/6)

    """
    Probes
    """
    # Probes for south turbine in parallel turbines
    # For each along wind location in wake, hub and apex height probed; for each along wind location in upstream, hub height probed
    # -3D, -1D, 1D, 2D, 4D
    southturb_probes = np.array(((916.725, 872.262, 90),
                       (1134.964, 998.262, 90),
                       (1353.202, 1124.262, 90), (1353.202, 1124.262, 153),
                       (1462.321, 1187.262, 90), (1462.321, 1187.262, 153),
                       (1680.56, 1313.26, 90), (1680.56, 1313.26, 153)))

    # Probes for offset turbines downwind turbine 7D behind upwind turbine
    # For each along wind location in wake, hub and apex height probed;
    # for each along wind location in upstream, hub height probed
    # -3D, -1D, 1D, 2D, 4D, 8D, 9D, 11D
    offsetturbs_probes = np.array(((788.857, 1089.422, 90),
                         (1007.096, 1215.422, 90),
                         (1225.334, 1341.422, 90), (1225.334, 1341.422, 153),
                         (1334.453, 1404.422, 90), (1334.453, 1404.422, 153),
                         (1552.692, 1530.422, 90), (1552.692, 1530.422, 153),
                         (1989.168, 1782.422, 90), (1989.168, 1782.422, 153),
                         (2098.288, 1845.422, 90), (2098.288, 1845.422, 153),
                         (2316.526, 1971.422, 90), (2316.526, 1971.422, 153)))

    # Probes for north turbine in parallel turbines
    # For each along wind location in wake, hub and apex height probed; for each along wind location in upstream,
    # hub height probed
    # -3D, -1D, 1D, 2D, 4D
    northturb_probes = np.array(((664.725, 1308.738, 90),
                       (882.964, 1434.738, 90),
                       (1101.202, 1560.738, 90), (1101.202, 1560.738, 153),
                       (1210.321, 1623.738, 90), (1210.321, 1623.738, 153),
                       (1428.56, 1749.738, 90), (1428.56, 1749.738, 153)))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(southturb_probes[:, 0], southturb_probes[:, 1], southturb_probes[:, 2])
    # ax.scatter(offsetturbs_probes[:, 0], offsetturbs_probes[:, 1], offsetturbs_probes[:, 2])
    # ax.scatter(northturb_probes[:, 0], northturb_probes[:, 1], northturb_probes[:, 2])

    """
    Plots
    """
    list_x2d, list_y2d = (xrefine1, xrefine2, xrefine3, xrefine4), (yrefine1, yrefine2, yrefine3, yrefine4)
    frameplot = BaseFigure3D(list_x2d, list_y2d, show=show, save=save, figdir=figdir, viewangle=viewangle, figwidth=figwidth, name=name, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    frameplot.initializeFigure()

    # # Whole domain
    # frameplot.axes[0].plot(xdomain, ydomain, zdomain, zorder = 10., linestyle = ':', color = frameplot.colors[2], alpha = alpha - 0.25)

    frameplot.axes.scatter(southturb_probes[:, 0], southturb_probes[:, 1], southturb_probes[:, 2], label='Probes ParTurb', c=frameplot.colors[0], linewidths=0, zorder=1.2, depthshade=scatter_depthshade, s=markersize)
    frameplot.axes.scatter(offsetturbs_probes[:, 0], offsetturbs_probes[:, 1], offsetturbs_probes[:, 2], label='Probes SeqTurb', c=frameplot.colors[1], linewidths=0, zorder=2.2, depthshade=scatter_depthshade, s=markersize)
    frameplot.axes.scatter(northturb_probes[:, 0], northturb_probes[:, 1], northturb_probes[:, 2], c=frameplot.colors[0], linewidths=0, zorder=1.3, depthshade=scatter_depthshade, s=markersize)

    frameplot.axes.plot(xrefine1, yrefine1, zrefine1, zorder=1.1, label='Refine2 ParTurb', linestyle='-', color=frameplot.colors[0], alpha=alpha)
    frameplot.axes.plot(xrefine2, yrefine2, zrefine2, zorder=1., label='Refine1 ParTurb', linestyle=':', color=frameplot.colors[0], alpha=alpha)
    frameplot.axes.plot(xrefine3, yrefine3, zrefine3, zorder=2.1, label='Refine2 SeqTurb', linestyle='-', color=frameplot.colors[1], alpha=alpha)
    frameplot.plot = frameplot.axes.plot(xrefine4, yrefine4, zrefine4, zorder=2., label='Refine1 SeqTurb', linestyle=':', color=frameplot.colors[1], alpha=alpha)

    frameplot.finalizeFigure(show_cbar=False, tight_layout=False)




