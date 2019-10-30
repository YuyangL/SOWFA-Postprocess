import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numba import jit, njit
from Utilities import timer
from PlottingTool import BaseFigure3D

case = 'both'  # 'ParTurb' 'Offset'
domainSize = (3000, 3000, 1000)
origin = (0, 0, 0)

figDir = '/media/yluan'
name = 'ProbesFront'
figWidth = 'half'
xLabel = r'$x$ [m]'
yLabel = r'$y$ [m]'
zLabel = r'$z$ [m]'
legLoc, alpha = 'upper center', 0.5
markerSize, scatterDepthShade = 10, True
viewAngle = (15, -150)
show, save = True, True

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

if case is 'both':
    # Parallel turbine inner refinement frame
    xRefine1, yRefine1, zRefine1 = generateFrameCoordinates(origin = (1120.344, 771.583, 0), frame_size = (882, 882, 279), rotation = np.pi/6)

    # Parallel turbine outer refinement frame
    xRefine2, yRefine2, zRefine2 = generateFrameCoordinates(origin = (1074.225, 599.464, 0), frame_size = (1134, 1134, 405), rotation = np.pi/6)

    # Offset turbine inner refinement frame
    xRefine3, yRefine3, zRefine3 = generateFrameCoordinates(origin = (994.344, 989.821, 0),
                                                            frame_size = (1764, 378, 279), rotation = np.pi/6)

    # Offset turbine outer refinement frame
    xRefine4, yRefine4, zRefine4 = generateFrameCoordinates(origin = (948.225, 817.702, 0),
                                                            frame_size = (1890, 630, 405), rotation = np.pi/6)

    """
    Probes
    """
    # Probes for south turbine in parallel turbines
    # For each along wind location in wake, hub and apex height probed; for each along wind location in upstream, hub height probed
    # -3D, -1D, 1D, 2D, 4D
    southTurbProbes = np.array(((916.725, 872.262, 90),
                       (1134.964, 998.262, 90),
                       (1353.202, 1124.262, 90), (1353.202, 1124.262, 153),
                       (1462.321, 1187.262, 90), (1462.321, 1187.262, 153),
                       (1680.56, 1313.26, 90), (1680.56, 1313.26, 153)))

    # Probes for offset turbines downwind turbine 7D behind upwind turbine
    # For each along wind location in wake, hub and apex height probed;
    # for each along wind location in upstream, hub height probed
    # -3D, -1D, 1D, 2D, 4D, 8D, 9D, 11D
    offsetTurbsProbes = np.array(((788.857, 1089.422, 90),
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
    northTurbProbes = np.array(((664.725, 1308.738, 90),
                       (882.964, 1434.738, 90),
                       (1101.202, 1560.738, 90), (1101.202, 1560.738, 153),
                       (1210.321, 1623.738, 90), (1210.321, 1623.738, 153),
                       (1428.56, 1749.738, 90), (1428.56, 1749.738, 153)))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(southTurbProbes[:, 0], southTurbProbes[:, 1], southTurbProbes[:, 2])
    # ax.scatter(offsetTurbsProbes[:, 0], offsetTurbsProbes[:, 1], offsetTurbsProbes[:, 2])
    # ax.scatter(northTurbProbes[:, 0], northTurbProbes[:, 1], northTurbProbes[:, 2])

    """
    Plots
    """
    listX2D, listY2D = (xRefine1, xRefine2, xRefine3, xRefine4), (yRefine1, yRefine2, yRefine3, yRefine4)
    framePlot = BaseFigure3D(listX2D, listY2D, show = show, save = save, figDir = figDir, viewAngles = viewAngle, figWidth = figWidth, name = name, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel)

    framePlot.initializeFigure()

    # # Whole domain
    # framePlot.axes[0].plot(xdomain, ydomain, zdomain, zorder = 10., linestyle = ':', color = framePlot.colors[2], alpha = alpha - 0.25)

    framePlot.axes[0].scatter(southTurbProbes[:, 0], southTurbProbes[:, 1], southTurbProbes[:, 2], label = 'Probes ParTurb', c = framePlot.colors[0], linewidths = 0, zorder = 1.2, depthshade = scatterDepthShade, s = markerSize)
    framePlot.axes[0].scatter(offsetTurbsProbes[:, 0], offsetTurbsProbes[:, 1], offsetTurbsProbes[:, 2], label = 'Probes SeqTurb', c = framePlot.colors[1], linewidths = 0, zorder = 2.2, depthshade = scatterDepthShade, s = markerSize)
    framePlot.axes[0].scatter(northTurbProbes[:, 0], northTurbProbes[:, 1], northTurbProbes[:, 2], c = framePlot.colors[0], linewidths = 0, zorder = 1.3, depthshade = scatterDepthShade, s = markerSize)

    framePlot.axes[0].plot(xRefine1, yRefine1, zRefine1, zorder = 1.1, label = 'Refine2 ParTurb', linestyle = '-', color = framePlot.colors[0], alpha = alpha)
    framePlot.axes[0].plot(xRefine2, yRefine2, zRefine2, zorder = 1., label = 'Refine1 ParTurb', linestyle = ':', color = framePlot.colors[0], alpha = alpha)
    framePlot.axes[0].plot(xRefine3, yRefine3, zRefine3, zorder = 2.1, label = 'Refine2 SeqTurb', linestyle = '-', color = framePlot.colors[1], alpha = alpha)
    framePlot.plot = framePlot.axes[0].plot(xRefine4, yRefine4, zRefine4, zorder = 2., label = 'Refine1 SeqTurb', linestyle = ':', color = framePlot.colors[1], alpha = alpha)

    framePlot.finalizeFigure(showCbar = False, tightLayout = True, legLoc = legLoc)




