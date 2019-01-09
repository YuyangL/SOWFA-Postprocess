import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.mlab as mlab
from scipy.interpolate import griddata

caseName = 'ALM_N_H'
# './ABL_N_H/Slices/20000.9038025/U_alongWind_Slice.raw'
data = np.genfromtxt('I:/SOWFA Data/' + caseName + '/Slices/20500.9078025/U_alongWind_Slice.raw', skip_header = 2)

x, y, z = data[:, 0], data[:, 1], data[:, 2]
pointsXZ = np.vstack((data[:, 0], data[:, 2])).T
u, v, w = data[:, 3], data[:, 4], data[:, 5]
# U
UmagSlice = np.zeros((data.shape[0], 1))
for i, row in enumerate(data):
    if np.nan in row:
        print(row)
    UmagSlice[i] = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2)


grid_x, grid_z = np.mgrid[x.min():x.max():1500j, z.min():z.max():500j]
grid_y, _ = np.mgrid[y.min():y.max():1500j, z.min():z.max():500j]

Uinterp = griddata(pointsXZ, UmagSlice.ravel(), (grid_x, grid_z), method = 'cubic')



from PlottingTool2 import Plot2D_InsetZoom, PlotSurfaceSlices3D

# myplot = Plot2D_InsetZoom(grid_x, grid_z, zoomBox = (1000, 2500, 0, 500), z2D = Uinterp, equalAxis = True, name = caseName + '_slice', figDir = 'R:/')
#
# myplot.initializeFigure()
# myplot.plotFigure(contourLvl = 100)
# myplot.finalizeFigure()

myplot2 = PlotSurfaceSlices3D(grid_x, grid_y, grid_z, Uinterp, name = caseName + '_3d', figDir = 'R:/', xLim = (0, 3000), yLim = (0, 3000), zLim = (0, 1000), show = False, xLabel = r'\textit{x} [m]', yLabel = r'\textit{z} [m]', zLabel = r'\textit{U} [m/s]')
myplot2.initializeFigure()
myplot2.plotFigure()
myplot2.finalizeFigure()



# from PlottingTool import plot2D, plot2DWithInsetZoom
# # plot2D(grid_x, grid_z, z2D = Uinterp, contourLvl = 10, equalAxis = True)
# plot2DWithInsetZoom(grid_x, grid_z, zoomBox = (1000, 2500, 0, 500), z2D = Uinterp, contourLvl = 100, equalAxis = True, name = caseName, xLabel = r'\textit{x} [m]', yLabel = r'\textit{z} [m]', zLabel = r'\textit{U} [m/s]')
#
# # fig, ax = plt.subplots(1, 1, num = 'asf')
# # plt.contourf(grid_x, grid_z, Uinterp, 100, extend = 'both')
# # # plt.axis('equal')
# # ax.set_aspect('equal', 'box')
# # plt.xlim((0, 3000))
# # plt.ylim((0, 1000))
# # plt.tight_layout()
#
# # plt.figure()
# # triang = tri.Triangulation(x, z)
# # plt.tricontourf(x, z, UmagSlice.ravel(), 20)
# # plt.colorbar()
# # plt.axis('equal')
