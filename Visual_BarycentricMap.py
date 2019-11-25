import time as t
import sys
# See https://github.com/YuyangL/TurbulenceMachineLearning
sys.path.append('/home/yluan/Documents/ML/TurbulenceMachineLearning')
from PlottingTool import PlotImageSlices3D, BaseFigure, Plot2D_InsetZoom, PlotContourSlices3D, plotTurbineLocations, PlotSurfaceSlices3D
from SliceData import *
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor
try:
    import PostProcess_AnisotropyTensor as PPAT
except ImportError:
    raise ImportError('\nNo module named PostProcess_AnisotropyTensor. Check SetupPython.py and run'
                      '\n[python SetupPython.py build_ext --inplace]')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy import ndimage
import pickle
import numpy as np
from DataBase import *
from copy import copy
# import visvis as vv
from Utility import interpolateGridData, rotateData
from matplotlib.patches import Circle, PathPatch


"""
User Inputs
"""
# Specify case time, usually "latestTime"
time = 'latestTime'  # str(float/int) or 'latestTime'
# Top absolute dir of case
# casedir = '/media/yluan'  # str
casedir = '/media/yluan'  # str
# Case name
casename = 'ALM_N_H_ParTurb2'  # str
# casename = 'RANS/N_H_OneTurb_Simple_ABL'  # str
# Flow Reynolds stress property, usually named 'uuPrime2' for LES or 'Rij' for RANS
property = 'uuPrime2'  # str
# property = 'Rij'  # str
# Slice names
slicenames = ('oneDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines')
# slicenames = ('threeDdownstreamTurbines', 'fiveDdownstreamTurbines', 'sevenDdownstreamTurbines')
# slicenames = ('hubHeight', 'quarterDaboveHub', 'turbineApexHeight')  # list(str)
# Subscript for the slice names, usually 'Slice'
slicenames_sub = 'Slice'  # str
# Result folder name
result_folder = 'Result'  # str
# Counter-clockwise orientation of x-axis in x-y plane, in case of angled flow direction [rad or deg]
rot_z = np.pi/6.  # float
# Turbine radius [m]
r = 63.  # float
# Whether to save processed slice data
save_data = False  # bool
# Whether rotate data by rot_z to align with flow dir
rotate_data = True  # bool
# Height of the horizontal slices, only used for 3D horizontal slices plot
horslice_offsets = (90., 121.5, 153.)
horslice_offsets2 = ((90., 90.), (121.5, 121.5), (153., 153.))


"""
Plot Settings
"""
# Which type(s) of plot to make
plot_type = '3D'  # '2D', '3D', 'all', '*'
# What value to plot, anisotropy tensor bij, Reynolds stress Rij, barycentric map, or all
plot_property = 'bijbary'  # 'bij', 'rij', 'bary', '*'
# Total number cells intended to plot via interpolation, usually 1e5
target_meshsize = 1e5  # int or float
# Number of contours for horizontal slices plot
contour_lvl = 200
# Interpolation method, usually linear
interp_method = 'linear'  # 'nearest', 'linear', 'cubic'
# Whether plot barycentric map example
show_baryexample = True  # bool
# Barycentric map color offset and exponent, usually 0.65 and 5
c_offset, c_exp = 0.65, 5.  # float, float
# Barycentric map example figure name
baryexample_name = 'barycentric_colormap'  # str
# Save figure format
ext = 'png'  # 'png', 'eps', 'jpg', 'pdf'
# Whether show and/or save figure
show, save = False, True  # bool, bool
# DPI of figure
dpi = 500  # int


"""
Process User Inputs
"""
# Ensure slicenames is a tuple if only 1 slice
slicenames = (slicenames,) if isinstance(slicenames, str) else slicenames
# Confined region auto definition
# If 1 turbine case, usually contains "OneTurb" in casename
if 'OneTurb' in casename:
    # For front view vertical slices, usually either "oneDupstream*" or "threeDdownstream*" is forefront slice
    if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        # Read coor info from database
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = OneTurb('vert')
        # If "threeDdownstream*" is 1st slice, then there're only 3 slices in total:
        # "threeDdownstream*", "fiveDdownstream", "sevenDdownstream"
        if 'threeDdownstream' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[3:]
        else:
            turb_centers_frontview = turb_centers_frontview[:3]
    # For horizontal slices, usually "hubHeight*" or "groundHeight*" is bottom slice
    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        turb_borders, turb_centers_frontview, confinebox, _ = OneTurb('hor')

# Else if parallel turbines, usually contains "ParTurb" in casename
elif 'ParTurb' in casename:
    if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
                                           'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'oneDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines',
                                           'threeDdownstreamTurbines', 'fiveDdownstreamTurbines',
                                           'sevenDdownstreamTurbines')
    if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        # Read coor info from database
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = ParTurb('vert')
        if 'threeDdownstream' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[6:]
        else:
            turb_centers_frontview = turb_centers_frontview[:6]

    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        turb_borders, turb_centers_frontview, confinebox, _ = ParTurb('hor')

# Automatic view angle and figure settings, only for 3D plots
# For front view vertical plots
if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
    if 'OneTurb' in casename:
        view_angle = (20, -80) if 'one' in slicenames[0] else (20, -95)
        # Equal axis option and figure width
        equalaxis, figwidth = True, 'half'
    elif 'ParTurb' in casename:
        view_angle = (20, -80) if 'one' in slicenames[0] else (20, -90)
        # Equal axis option and figure width
        equalaxis, figwidth = True, 'half'
# For horizontal plots
elif 'groundHeight' in slicenames[0] or 'hubHeight' in slicenames[0]:
    view_angle = (25, -115)
    equalaxis, figwidth = False, 'half'
# Default settings for other cases
else:
    view_angle = (20, -100)
    equalaxis, figwidth = True, 'full'

# Unify plot_type user inputs
if plot_type in ('2D', '2d'):
    plot_type = '2D'
elif plot_type in ('3D', '3d'):
    plot_type = '3D'
elif plot_type in ('all', 'All', '*'):
    plot_type = '*'

# Value limit for plotting
# Usually (-1/2, 2/3) for anisotropy tensor bij
bij_lim = (-0.5, 2/3.)
# Usually None for Reynolds stress Rij
rij_lim = None #(-1., 1.)
# Value labels for bij and Rij
# If Rij is called uuPrime2, it's from LES and is statistically averaged
if 'uuPrime2' in property:
    bij_label = (r'$\langle b_{11} \rangle$ [-]', r'$\langle b_{12} \rangle$ [-]', r'$\langle b_{13} \rangle$ [-]',
                 r'$\langle b_{22} \rangle$ [-]', r'$\langle b_{23} \rangle$ [-]',
                 r'$\langle b_{33} \rangle$ [-]')
    rij_label = [r"$\langle u'u' \rangle$", r"$\langle u'v' \rangle$", r"$\langle u'w' \rangle$",
                 r"$\langle v'v' \rangle$", r"$\langle v'w' \rangle$",
                 r"$\langle w'w' \rangle$"]
# Else if in RANS, Rij is not averaged
else:
    bij_label = ('$b_{11}$ [-]', '$b_{12}$ [-]', '$b_{13}$ [-]',
                 '$b_{22}$ [-]', '$b_{23}$ [-]',
                 '$b_{33}$ [-]')
    rij_label = [r"$u'u'$", r"$u'v'$", r"$u'w'$",
                 r"$v'v'$", r"$v'w'$",
                 r"$w'w'$"]
    
# Add unit to Rij
for i in range(6):
    rij_label[i] += ' [m$^2$/s$^2$]'

# Auto deg to rad unit conversion
if rot_z > np.pi/2.: rot_z *= np.pi/180.
make_anisotropic = False if property == 'bij' else True


"""
Read Slice Data
"""
# Initialize case slice instance
case = SliceProperties(time=time, casedir=casedir, casename=casename, rot_z=rot_z, result_folder=result_folder)
# Read slices and store them in case.slice_* dict
case.readSlices(properties=property, slicenames=slicenames, slicenames_sub=slicenames_sub)



# # Initialize lists to store multiple slices coor and value data
# list_x, list_y, list_z = [], [], []
# list_rgb, list_bij, list_rij = [], [], []
# # Go through specified slices
# for i, slicename in enumerate(case.slicenames):
#     """
#     Process Uninterpolated Anisotropy Tensor
#     """
#     # Retrieve Rij of this slice
#     rij = case.slices_val[slicename]
#     rij = expandSymmetricTensor(rij).reshape((-1, 3, 3))
#     # Rotate Rij if requested, rotateData only accept full matrix form
#     if rotate_data: rij = rotateData(rij, anglez=rot_z)
#     rij = contractSymmetricTensor(rij)
#     rij_tmp = rij.copy()
#     # Get bij from Rij and its corresponding eigenval and eigenvec
#     bij, eig_val, eig_vec = processReynoldsStress(rij_tmp, realization_iter=0, make_anisotropic=make_anisotropic, to_old_grid_shape=False)
#     # bij was (n_samples, 3, 3) contract it
#     bij = contractSymmetricTensor(bij)
#     # Get barycentric map coor and normalized RGB
#     xy_bary, rgb = getBarycentricMapData(eig_val, c_offset=c_offset, c_exp=c_exp)
#
#
#     """
#     Interpolation
#     """
#     # 1st coor is always x not matter vertical or horizontal slice
#     # 2nd coor is y if horizontal slice otherwise z, take appropriate confinebox limit
#     coor2_lim = confinebox[i][2:4] if case.slices_orient[slicename] == 'horizontal' else confinebox[i][4:]
#     # Selective interpolation upon request
#     if 'bary' in plot_property or '*' in plot_property:
#         x_mesh, y_mesh, z_mesh, rgb_mesh = case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0], case.slices_coor[slicename][:, 1], case.slices_coor[slicename][:, 2], rgb,
#                                                                                     slice_orient=case.slices_orient[slicename], target_meshsize=target_meshsize,
#                                                                                     # No interpolation as it gives undefined colors in my barycentric map
#                                                                                     interp_method='nearest', confinebox=confinebox[i])
#         if plot_type in ('3D', '*'): list_rgb.append(rgb_mesh)
#         if save_data: pickle.dump(rgb_mesh, open(case.result_path + slicename + '_BaryRGB.p', 'wb'))
#
#     if 'bij' in plot_property or '*' in plot_property:
#         x_mesh, y_mesh, z_mesh, bij_mesh = case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0],
#                                                                                     case.slices_coor[slicename][:, 1],
#                                                                                     case.slices_coor[slicename][:, 2],
#                                                                                     bij,
#                                                                                     slice_orient=case.slices_orient[
#                                                                                         slicename],
#                                                                                     target_meshsize=target_meshsize,
#                                                                                     interp_method=interp_method,
#                                                                                     confinebox=confinebox[i])
#         if plot_type in ('3D', '*'): list_bij.append(bij_mesh)
#         if save_data: pickle.dump(bij_mesh, open(case.result_path + slicename + '_bij.p', 'wb'))
#
#     if 'rij' in plot_property or '*' in plot_property:
#         x_mesh, y_mesh, z_mesh, rij_mesh = case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0],
#                                                                                     case.slices_coor[slicename][:, 1],
#                                                                                     case.slices_coor[slicename][:, 2],
#                                                                                     rij,
#                                                                                     slice_orient=case.slices_orient[
#                                                                                         slicename],
#                                                                                     target_meshsize=target_meshsize,
#                                                                                     interp_method=interp_method,
#                                                                                     confinebox=confinebox[i])
#         if plot_type in ('3D', '*'): list_rij.append(rij_mesh)
#         if save_data: pickle.dump(rij_mesh, open(case.result_path + slicename + '_Rij.p', 'wb'))
#
#     # Append 2D mesh to a list for 3D plots
#     if plot_type in ('3D', '*'):
#         list_x.append(x_mesh)
#         list_y.append(y_mesh)
#         list_z.append(z_mesh)
#
#     if save_data:
#         pickle.dump(x_mesh, open(case.result_path + slicename + '_xmesh.p', 'wb'))
#         pickle.dump(y_mesh, open(case.result_path + slicename + '_ymesh.p', 'wb'))
#         pickle.dump(z_mesh, open(case.result_path + slicename + '_zmesh.p', 'wb'))
#
#
# """
# Plotting
# """
# if plot_type in ('3D', 'all'):
#     if case.slices_orient[slicename] == 'horizontal':
#         show_xylabel = (False, False)
#         show_zlabel = True
#         show_ticks = (False, False, True)
#     else:
#         show_xylabel = (True, True)
#         show_zlabel = False
#         show_ticks = (True, True, False)
#
#     # Barycentric map
#     if 'bary' in plot_property or '*' in plot_property:
#         figheight_multiplier = 2 if case.slices_orient[slicename] == 'horizontal' else 0.75
#         print(min(rgb_mesh.ravel()))
#         barymap_slice3d = PlotImageSlices3D(list_x=list_x, list_y=list_y, list_z=list_z, list_rgb=list_rgb,
#                                            name='barycentric_' + str(
#                 slicenames),
#                                            xlabel=r'$x$ [m]',
#                                   ylabel=r'$y$ [m]', zlabel = r'$z$ [m]', save=save, show=show,
#                                         figdir=case.result_path, viewangle=view_angle, figwidth=figwidth, figheight_multiplier=figheight_multiplier,
#                                             equalaxis=equalaxis)
#         barymap_slice3d.initializeFigure(constrained_layout=True)
#         barymap_slice3d.plotFigure()
#         plotTurbineLocations(barymap_slice3d, case.slices_orient[slicename], horslice_offsets, turb_borders, turb_centers_frontview)
#         barymap_slice3d.finalizeFigure(tight_layout=False, show_xylabel=(False,)*2, show_ticks=(False,)*3, show_zlabel=False)
#         print(min(rgb_mesh.ravel()))
#
#
#     # bij
#     if 'bij' in plot_property or '*' in plot_property:
#         figheight_multiplier = 1.75 if case.slices_orient[slicename] == 'horizontal' else 1
#         ij = ['11', '12', '13', '22', '23', '33']
#         for i in range(6):
#             # Extract ij component from each slice's bij
#             list_bij_i = [bij_i[..., i] for _, bij_i in enumerate(list_bij)]
#             # If horizontal slices 3D plot
#             if case.slices_orient[slicename] == 'horizontal':
#                 bij_slice3d = PlotContourSlices3D(list_x, list_y, list_bij_i, horslice_offsets,
#                                                     name='b' + ij[i] + '_' + str(
#                                                             slicenames),
#                                                     xlabel=r'$x$ [m]',
#                                                     ylabel=r'$y$ [m]', zlabel=r'$z$ [m]', val_label=bij_label[i],
#                                                   cbar_orient='vertical',
#                                                   save=save, show=show,
#                                                     figdir=case.result_path, viewangle=view_angle, figwidth=figwidth,
#                                                     figheight_multiplier=figheight_multiplier,
#                                                   val_lim=bij_lim, equalaxis=equalaxis)
#             # Else if vertical front view slices 3D plot
#             else:
#                 bij_slice3d = PlotSurfaceSlices3D(list_x, list_y, list_z, list_bij_i,
#                                                   xlabel='$x$ [m]', ylabel='$y$ [m]', zlabel='$z$ [m]', val_label=bij_label[i],
#                                                   name='b' + ij[i] + '_' + str(
#                                                           slicenames),
#                                                   save=save, show=show,
#                                                   figdir=case.result_path, viewangle=view_angle, figwidth=figwidth,
#                                                   equalaxis=equalaxis, cbar_orient='horizontal', val_lim=bij_lim)
#
#             bij_slice3d.initializeFigure()
#             bij_slice3d.plotFigure(contour_lvl=contour_lvl)
#             plotTurbineLocations(bij_slice3d, case.slices_orient[slicename], horslice_offsets, turb_borders,
#                                  turb_centers_frontview)
#             bij_slice3d.finalizeFigure(show_xylabel=show_xylabel, show_zlabel=show_zlabel, show_ticks=show_ticks)
#
#     # Rij
#     if 'rij' in plot_property or '*' in plot_property:
#         figheight_multiplier = 1.75 if case.slices_orient[slicename] == 'horizontal' else 1
#         ij = ['11', '12', '13', '22', '23', '33']
#         for i in range(6):
#             # Extract ij component from each slice's bij
#             list_rij_i = [rij_i[..., i] for _, rij_i in enumerate(list_rij)]
#             if i in (0, 3, 5):
#                 for j in range(len(list_rij_i)):
#                     list_rij_i[j][list_rij_i[j] < 0.] = 0.
#
#             else:
#                 for j in range(len(list_rij_i)):
#                     list_rij_i[j][list_rij_i[j] < -1.] = -1.
#
#             # If horizontal slices 3D plot
#             if case.slices_orient[slicename] == 'horizontal':
#                 rij_slice3d = PlotContourSlices3D(list_x, list_y, list_rij_i, horslice_offsets,
#                                                   name='R' + ij[i] + '_' + str(
#                                                           slicenames),
#                                                   xlabel=r'$x$ [m]',
#                                                   ylabel=r'$y$ [m]', zlabel=r'$z$ [m]', val_label=rij_label[i],
#                                                   cbar_orient='vertical',
#                                                   save=save, show=show,
#                                                   figdir=case.result_path, viewangle=view_angle, figwidth=figwidth,
#                                                   figheight_multiplier=figheight_multiplier, equalaxis=equalaxis,
#                                                   val_lim=rij_lim)
#             # Else if vertical front view slices 3D plot
#             else:
#                 rij_slice3d = PlotSurfaceSlices3D(list_x, list_y, list_z, list_rij_i,
#                                                   xlabel='$x$ [m]', ylabel='$y$ [m]', zlabel='$z$ [m]',
#                                                   val_label=rij_label[i],
#                                                   name='R' + ij[i] + '_' + str(
#                                                           slicenames),
#                                                   save=save, show=show,
#                                                   figdir=case.result_path, viewangle=view_angle, figwidth=figwidth,
#                                                   equalaxis=equalaxis, cbar_orient='horizontal', val_lim=rij_lim)
#
#             rij_slice3d.initializeFigure()
#             rij_slice3d.plotFigure(contour_lvl=contour_lvl)
#             plotTurbineLocations(rij_slice3d, case.slices_orient[slicename], horslice_offsets, turb_borders,
#                                  turb_centers_frontview)
#             rij_slice3d.finalizeFigure(show_xylabel=show_xylabel, show_zlabel=show_zlabel, show_ticks=show_ticks)
#
# if save and plot_type in ('2D', '*'):
#     print('\n2D plot of {0} saved at {1}'.format(slicenames, case.result_path))




"""
Plot Barycentric Color Map If Requested
"""
if show_baryexample:
    xTriLim, yTriLim = (0, 1), (0, np.sqrt(3) / 2.)
    verts = (
        (xTriLim[0], yTriLim[0]), (xTriLim[1], yTriLim[0]), (np.mean(xTriLim), yTriLim[1]),
        (xTriLim[0], yTriLim[0]))
    triangle = Path(verts)

    xTri, yTri = np.mgrid[xTriLim[0]:xTriLim[1]:1000j, yTriLim[0]:yTriLim[1]:1000j]
    xyTri = np.transpose((xTri.ravel(), yTri.ravel()))
    mask = triangle.contains_points(xyTri)
    # mask = mask.reshape(xTri.shape).T
    xyTri = xyTri[mask]
    # xTri, yTri = np.ma.array(xTri, mask, dtype = bool), np.ma.array(yTri, mask)

    c3 = xyTri[:, 1] / yTriLim[1]
    c1 = xyTri[:, 0] - 0.5 * c3
    c2 = 1 - c1 - c3

    rgbVals_example = np.vstack((c1, c2, c3)).T
    # rgbValsNew = np.empty((c1.shape[0], 3))
    # Each 2nd dim is an RGB array of the 2D grid
    rgbValsNew_example = (rgbVals_example + c_offset) ** c_exp

    baryMap3D = griddata(xyTri, rgbValsNew_example, (xTri, yTri))
    baryMap3D[np.isnan(baryMap3D)] = 1.
    baryMap3D = ndimage.rotate(baryMap3D, 90)


    baryMap3D[baryMap3D[..., 0] <= 0.99] = [.3, .3, .3]

    baryMapExample = BaseFigure((None,), (None,), name=baryexample_name, figdir=case.result_path,
                                show=show, save=save)
    baryMapExample.initializeFigure()
    baryMapExample.axes.imshow(baryMap3D, origin='upper', aspect='equal', extent=(xTriLim[0], xTriLim[1],
                                                                                           yTriLim[0], yTriLim[1]))
    baryMapExample.axes.annotate(r'$\textbf{x}_{2c}$', (xTriLim[0], yTriLim[0]), (xTriLim[0] - 0.1, yTriLim[0]))
    baryMapExample.axes.annotate(r'$\textbf{x}_{3c}$', (np.mean(xTriLim), yTriLim[1]))
    baryMapExample.axes.annotate(r'$\textbf{x}_{1c}$', (xTriLim[1], yTriLim[0]))
    # baryMapExample.axes[0].get_yaxis().set_visible(False)
    # baryMapExample.axes[0].get_xaxis().set_visible(False)
    # baryMapExample.axes[0].set_axis_off()
    baryMapExample.axes.axis('off')
    plt.savefig(case.result_path + baryexample_name + '.' + ext, transparent=True, dpi=dpi)
    if save:
        print('\n{0} saved at {1}'.format(baryexample_name, case.result_path))

    # baryPlot = PlotSurfaceSlices3D(x2D, y2D, z2D, (0,), show = True, name = 'bary', figdir = 'R:', save = True)
    # baryPlot.cmapLim = (0, 1)
    # baryPlot.cmapNorm = rgb
    # # baryPlot.cmapVals = plt.cm.ScalarMappable(norm = rgb, cmap = None)
    # baryPlot.cmapVals = rgb
    # # baryPlot.cmapVals.set_array([])
    # baryPlot.plot = baryPlot.cmapVals
    # baryPlot.initializeFigure()
    # baryPlot.axes[0].plot_surface(x2D, y2D, z2D, cstride = 1, rstride = 1, facecolors = rgb, vmin = 0, vmax = 1, shade = False)
    # baryPlot.finalizeFigure()
    #
    #
    #
    #
    #
    # print('\nDumping values...')
    # pickle.dump(tensors, open(case.result_path + slicename + '_tensors.p', 'wb'))
    # pickle.dump(case.slicesCoor[slicename][:, 0], open(case.result_path + slicename + '_x.p', 'wb'))
    # pickle.dump(case.slicesCoor[slicename][:, 1], open(case.result_path + slicename + '_y.p', 'wb'))
    # pickle.dump(case.slicesCoor[slicename][:, 2], open(case.result_path + slicename + '_z.p', 'wb'))
    #
    # print('\nExecuting RGB_barycentric_colors_clean...')
    # import RBG_barycentric_colors_clean
    #
    #
    #
    #
    # # valsDecomp = case.mergeHorizontalComponents(valsDecomp)
