"""
Read and Visualize Horizontal/Vertical Slices in 2/3D
"""
import numpy as np
import os
from Utility import timer
from scipy.interpolate import griddata
from PlottingTool import Plot2D, Plot2D_InsetZoom, PlotSurfaceSlices3D, PlotContourSlices3D, pathpatch_translate, pathpatch_2d_to_3d, PlotImageSlices3D
try:
    import PostProcess_Tensor as PPT
except ImportError:
    raise ImportError('\nNo module named PostProcess_Tensor. Check setup.py and run'
                      '\npython setup.py build_ext --inplace')

try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch
import SliceData as PPSD
from DataBase import *
from copy import copy
from Utility import fieldSpatialSmoothing
from scipy.ndimage import gaussian_filter

"""
User Inputs
"""
time = '10000'  #'23243.2156219'
# time = 'latestTime'  #'23243.2156219'
casedir = '/media/yluan'
casename = 'RANS/N_H_ParTurb_LowZ_Rwall'  #'RANS/N_H_OneTurb_Simple_ABL'  #'URANS/N_H_OneTurb'  # 'ALM_N_H_ParTurb'
# casename = 'ALM_N_H_OneTurb'
# properties = ('kResolved', 'kSGSmean')
# properties = ('divDevR', 'divDevR_pred_TBDT', 'divDevR_pred_TBRF', 'divDevR_pred_TBAB', 'divDevR_pred_TBGB')
# properties = ('divDevR_blend', 'divDevR_pred_TBDT', 'divDevR_pred_TBRF', 'divDevR_pred_TBAB', 'divDevR_pred_TBGB')
properties = ('GAvg', 'G_pred_TBDT', 'G_pred_TBRF', 'G_pred_TBAB', 'G_pred_TBGB')
# properties = ('G', 'G_pred_TBDT', 'G_pred_TBRF', 'G_pred_TBAB', 'G_pred_TBGB')
properties = ('RGB_pred_TBDT', 'RGB_pred_TBRF', 'RGB_pred_TBAB', 'RGB_pred_TBGB')
properties = ('RGB',)
# slicenames = ('oneDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine')
# slicenames = ('threeDdownstreamTurbine', 'fiveDdownstreamTurbine', 'sevenDdownstreamTurbine')
slicenames = ('hubHeight', 'quarterDaboveHub', 'turbineApexHeight')
# slicenames = ('groundHeight', 'hubHeight', 'oneDaboveHubHeight')
# Subscript for the slice names
slicenames_sub = 'Slice'
# Height of the horizontal slices, only used for 3D horizontal slices plot
horslice_offsets = (90., 121.5, 153.)
# horslice_offsets = (0, 90, 216)
horslice_offsets2 = ((90., 90.), (121.5, 121.5), (153., 153.))
result_folder = 'Result'
# Orientation of x-axis in x-y plane, in case of angled flow direction
# Only used for values decomposition and confinebox
# Angle in rad and counter-clockwise
rot_z = np.pi/6.
# Turbine radius, only used for confinebox
r = 63
# For calculating total <epsilon> only
nu = 1e-5
filtering = True


"""
Plot Settings
"""
# Which type(s) of plot to make
plot_type = '3D'  # '2D', '3D', 'all'
# Total number cells intended to plot via interpolation
target_meshsize = 2e5
interp_method = 'linear'
# Number of contours, only for 2D plots or 3D horizontal slice plots
contour_lvl = 200
# Label of the property, could be overridden below
val_label = 'Data'
ext = 'png'
show, save = False, True
dpi = 500


"""
Process User Inputs
"""
# Ensure slicenames is a tuple
slicenames = (slicenames,) if isinstance(slicenames, str) else slicenames
# Confined region auto definition
if 'OneTurb' in casename:
    rotate_field = False
    # For rotor plane vertical slices
    if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = OneTurb('vert')
        if 'three' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[3:]
        else:
            turb_centers_frontview = turb_centers_frontview[:3]

    # For horizontal slices
    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        # Confinement for z doesn't matter since the slices are horizontal
        confinebox = ((800, 2400, 800, 2400, 0, 216),)*len(slicenames)
        turb_borders, turb_centers_frontview, confinebox, _ = OneTurb('hor', rotate_field=rotate_field)

elif 'ParTurb' in casename:
    rotate_field = True
    if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        # Read coor info from database
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = ParTurb('vert')
        if 'threeDdownstream' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[6:]
        else:
            turb_centers_frontview = turb_centers_frontview[:6]

    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        if 'Yaw' in casename:
            turb_borders, turb_centers_frontview, confinebox, _ = ParTurb_Yaw('hor', rotate_field=rotate_field)
        else:
            turb_borders, turb_centers_frontview, confinebox, _ = ParTurb('hor', rotate_field=rotate_field)

elif 'SeqTurb' in casename:
    rotate_field = False
    # For rotor plane vertical slices
    # TODO: update for vertical slices
    if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
        turb_borders, turb_centers_frontview, confinebox, confinebox2 = SeqTurb('vert')
        if 'three' in slicenames[0]:
            confinebox = confinebox2
            turb_centers_frontview = turb_centers_frontview[3:]
        else:
            turb_centers_frontview = turb_centers_frontview[:3]

    # For horizontal slices
    elif 'hubHeight' in slicenames[0] or 'groundHeight' in slicenames[0]:
        # Confinement for z doesn't matter since the slices are horizontal
        turb_borders, turb_centers_frontview, confinebox, _ = SeqTurb('hor', rotate_field=rotate_field)

else:
    turb_borders = ((99999,)*4,)
    turb_centers_frontview = ((99999,)*3,)*6
    confinebox = confinebox2 = [[5., 2995., 5., 2995., 5., 995.]]*10

# If you don't want confinement
# confinebox = confinebox2 = [[5., 2995., 5., 2995., 5., 995.]]*10

# Automatic view_angle and figure settings, only for 3D plots
if 'oneDupstream' in slicenames[0] or 'threeDdownstream' in slicenames[0]:
    view_angle = (20, -80) if 'one' in slicenames[0] else (20, -95)
    equalaxis, figwidth = True,  'half'
elif 'groundHeight' in slicenames[0] or 'hubHeight' in slicenames[0]:
    view_angle = (25, -115)
    equalaxis, figwidth = False, 'half'
else:
    view_angle = (20, -100)
    equalaxis, figwidth = True, 'full'

# Unify plot_type user inputs
if plot_type in ('2D', '2d'):
    plot_type = '2D'
elif plot_type in ('3D', '3d'):
    plot_type = '3D'
elif plot_type in ('all', 'All', '*'):
    plot_type = 'all'

if 'U' in properties[0]:
    val_lim = (0, 12) if 'HiSpeed' not in casename else (0, 12)
    val_lim_z = (-2, 2)
    if 'ALM' in casename:
        val_label = [r'$\langle \tilde{U} \rangle_\mathrm{hor}$ [m/s]', r'$\langle \tilde{u}_z \rangle$ [m/s]']
    elif 'ABL' in casename:
        val_label = [r'$\tilde{U}_\mathrm{hor}$ [m/s]', r'$\tilde{u}_z$ [m/s]']
    else:
        val_label = [r'$\langle U \rangle_\mathrm{hor}$ [m/s]', r'$\langle u_z \rangle$ [m/s]']

elif 'k' in properties[0]:
    if 'HiSpeed' not in casename:
        val_lim = (0., 3.) if 'SGS' not in properties[0] else (0., 0.15)
    else:
        val_lim = (0., 3.) if 'SGS' not in properties[0] else (0., 0.15)

    val_lim_z = None

    if 'Resolved' in properties[0]:
        val_label = (r'$\langle k_\mathrm{resolved} \rangle$ [m$^2$/s$^2$]',) if len(properties) == 1 else (r'$\langle k \rangle$ [m$^2$/s$^2$]',)
    elif 'SGS' in properties[0]:
        val_label = (r'$\langle k_\mathrm{SGS} \rangle$ [m$^2$/s$^2$]',)
    else:
        val_label = (r'$\langle k \rangle$ [m$^2$/s$^2$]',)

elif 'uuPrime2' in properties[0] or 'R' in properties[0] and 'div' not in properties[0]:
    val_lim = (-0.5, 2/3.)
    val_lim_z = None
    val_label = (r"$\langle u'u' \rangle$ [-]", r"$\langle u'v' \rangle$ [-]", r"$\langle u'w' \rangle$ [-]",
                 r"$\langle v'v' \rangle$ [-]", r"$\langle v'w' \rangle$ [-]",
                                                r"$\langle w'w' \rangle$ [-]")
elif "epsilon" in properties[0]:
    if 'HiSpeed' not in casename:
        val_lim = (0., 0.012) if 'SGS' not in properties[0] else (0., 0.012)
    else:
        val_lim = (0., 0.04)

    val_lim = (0, 0.015)

    val_lim_z = None
    if 'SGS' in properties[0]:
        val_label = (r'$\langle \epsilon_{\mathrm{SGS}} \rangle$ [m$^2$/s$^3$]',) if 'mean' in properties[0] else (r'$\epsilon_{\mathrm{SGS}}$ [m$^2$/s$^3$]',)
    elif 'Resolved' in properties[0]:
        val_label = (r'$\langle \epsilon_{\mathrm{resolved}} \rangle$ [m$^2$/s$^3$]',)
    else:
        val_label = (r'$\langle \epsilon \rangle$ [m$^2$/s$^3$]',)

elif 'G' in properties[0] and 'RGB' not in properties[0]:
    val_lim = (0., .15) if 'LowZ' not in casename else (0., .15)
    val_lim_z = None
    val_label = (r'$\langle G \rangle$ [m$^2$/s$^3$]',)
elif 'divDevR' in properties[0]:
    val_lim = (0., .1)
    # val_lim_z = (-0.08, 0.07) if 'HiSpeed' not in casename else (-0.11, 0.1)
    val_lim_z = (-0.05, 0.05) if 'HiSpeed' not in casename else (-0.05, 0.05)
    val_label = (r'$\langle \nabla \cdot R_{ij}^D \rangle_\mathrm{hor}$ [m/s$^2$]', r'$\langle \nabla \cdot R_{ij}^D \rangle_z$ [m/s$^2$]')
else:
    val_lim = None
    val_lim_z = None
    val_label = ('data',)

"""
Read Slice Data
"""
for i0 in range(len(properties)):
    # Initialize case
    case = PPSD.SliceProperties(time=time, casedir=casedir, casename=casename, rot_z=rot_z, result_folder=result_folder)
    # Read slices
    case.readSlices(properties=properties[i0], slicenames=slicenames, slicenames_sub=slicenames_sub)

    list_x2d, list_y2d, list_z2d, list_val3d, list_val3d_z = [], [], [], [], []
    # Go through specified slices and flow properties
    for i in range(len(case.slicenames)):
        # for i, slicename in enumerate(case.slicenames):
        slicename = case.slicenames[i]
        vals2d = case.slices_val[slicename]
        # If kResolved and kSGSmean in properties, get total kMean
        if 'kResolved' in properties and 'kSGSmean' in properties:
            print(' Calculating total <k> for {}...'.format(slicenames[i]))
            slicename2 = case.slicenames[i + len(slicenames)]
            vals2d += case.slices_val[slicename2]
        elif 'Rij' in properties or 'uuPrime2' in properties:
            print('Calculating Barycentric map...')
            # k = .5*(vals2d[:, 0] + vals2d[:, 3] + vals2d[:, 5])
            # for j in range(6):
            #     vals2d[:, j] = vals2d[:, j]/(2.*k) - 1/3. if j in (0, 3, 5) else vals2d[:, j]/(2.*k)
            #     bij =

            _, eigval, _ = PPT.processReynoldsStress(vals2d, make_anisotropic=True)
            # eigval = eigval.reshape((len(eigval), 1, 3))
            _, vals2d = PPT.getBarycentricMapData(eigval)
            # vals2d = vals2d.reshape((len(vals2d), 3))
            vals2d[vals2d > 1.] = 1.



        # # Else if epsilonSGSmean and nuSGSmean in properties then get total epsilonMean
        # # By assuming isotropic homogeneous turbulence and
        # # <epsilon> = <epsilonSGS>/(1 - 1/(1 + <nuSGS>/nu))
        # elif 'epsilonSGSmean' in properties and 'nuSGSmean' in properties:
        #     print(' Calculating total <epsilon> for {}...'.format(slicenames[i]))
        #     slicename2 = case.slicenames[i + len(slicenames)]
        #     vals2d_2 = case.slices_val[slicename2]
        #     # Determine which vals2d is epsilonSGSmean or nuSGSmean
        #     nusgs_mean, epsilonSGSmean = (vals2d, vals2d_2) if 'epsilonSGSmean' in slicename2 else (vals2d_2, vals2d)
        #     # Calculate epsilonMean
        #     vals2d = case.calcSliceMeanDissipationRate(epsilonSGSmean = epsilonSGSmean, nusgs_mean = nusgs_mean, nu = nu)

        # Interpolation an
        if 'pred' in properties[i0] and filtering:
            # x2d, y2d, z2d, vals3d = case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0],
            #                                                                  case.slices_coor[slicename][:, 1],
            #                                                                  case.slices_coor[slicename][:, 2], vals2d,
            #                                                                  slice_orient=case.slices_orient[slicename],
            #                                                                  rot_z=rot_z,
            #                                                                  target_meshsize=target_meshsize,
            #                                                                  interp_method=interp_method,
            #                                                                  confinebox=confinebox[i])
            #
            # vals3d = gaussian_filter(vals3d, sigma=5.)
            if rotate_field:
                print('\nRotating x and y...')
                x_tmp, y_tmp = case.slices_coor[slicename][:, 0].copy(), case.slices_coor[slicename][:, 1].copy()
                x = np.cos(rot_z)*x_tmp + np.sin(rot_z)*y_tmp
                y = np.cos(rot_z)*y_tmp - np.sin(rot_z)*x_tmp
                print('\nPerforming 2D Gaussian filtering for {}...'.format(properties[i0]))
            else:
                x, y = case.slices_coor[slicename][:, 0], case.slices_coor[slicename][:, 1]

            val_lim2 = (val_lim[0] - 2.*(val_lim[1] - val_lim[0]),
                        val_lim[1] + 2.*(val_lim[1] - val_lim[0]))
            x2d, y2d, _, vals3d = fieldSpatialSmoothing(vals2d, x, y, xlim=tuple(confinebox[i][:2]), ylim=tuple(confinebox[i][2:4]), mesh_target=target_meshsize, interp_method=interp_method, val_bnd=val_lim2)
            z = case.slices_coor[slicename][:, 2]
            _, z2d = np.mgrid[0:1:x2d.shape[0]*1j, z.min():z.max():x2d.shape[1]*1j]
        else:
            rot_z_tmp = 0 if not rotate_field else rot_z
            x2d, y2d, z2d, vals3d = case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0], case.slices_coor[slicename][:, 1], case.slices_coor[slicename][:, 2], vals2d,
                                                                             slice_orient=case.slices_orient[slicename], rot_z=rot_z_tmp,
                                                                             target_meshsize=target_meshsize,
                                                                             interp_method=interp_method,
                                                                             confinebox=confinebox[i])

            # Flatten if vals3d only have one component like a scalar field
            if vals3d.shape[2] == 1: vals3d = vals3d.reshape((vals3d.shape[0], vals3d.shape[1]))

        # Calculate magnitude if U
        if 'U' in properties[0] or 'divDevR' in properties[0]:
            vals3d_hor = np.sqrt(vals3d[:, :, 0]**2 + vals3d[:, :, 1]**2)
            vals3d_z =  vals3d[:, :, 2]
        else:
            vals3d_hor = vals3d
            vals3d_z = None

        # Append 2D mesh to a list for 3D plots
        if plot_type in ('3D', 'all'):
            list_x2d.append(x2d)
            list_y2d.append(y2d)
            list_z2d.append(z2d)
            if 'RGB' not in properties[0] and 'Rij' not in properties and 'uuPrime2' not in properties:
                vals3d_hor = np.nan_to_num(vals3d_hor)
                vals3d_hor[vals3d_hor > val_lim[1]] = val_lim[1]
                vals3d_hor[vals3d_hor < val_lim[0]] = val_lim[0]
                if vals3d_z is not None:
                    vals3d_z = np.nan_to_num(vals3d_z)
                    vals3d_z[vals3d_z > val_lim_z[1]] = val_lim_z[1]
                    vals3d_z[vals3d_z < val_lim_z[0]] = val_lim_z[0]

            list_val3d.append(vals3d_hor)
            list_val3d_z.append(vals3d_z)

        # Determine the unit along the vertical slice since it's angled, only for 2D plots of vertical slices
        if case.slices_orient[slicename] == 'vertical':
            # # If angle from x-axis is 45 deg or less
            # if lx >= ly:
            #     rot_z = np.arctan(lx/ly)
            if confinebox is None:
                lx = np.max(x2d) - np.min(x2d)
                ly = np.max(y2d) - np.min(y2d)
            else:
                lx = confinebox[i][1] - confinebox[i][0]
                ly = confinebox[i][3] - confinebox[i][2]

            r2d = np.linspace(0, np.sqrt(lx**2 + ly**2), x2d.shape[0])

        # Break if i finishes all kResolved or kSGSmean
        if 'kResolved' in properties and 'kSGSmean' in properties and i == (len(slicenames) - 1):
            break
        elif 'epsilonSGSmean' in properties and 'nuSGSmean' in properties and i == (len(slicenames) - 1):
            break


        """
        Plotting
        """
        xlabel, ylabel = (r'$x$ [m]', r'$y$ [m]') \
            if case.slices_orient[slicename] == 'vertical' else \
            (r'$x$ [m]', r'$y$ [m]')
        # Figure name
        if 'kResolved' in properties and 'kSGSmean' in properties:
            figname = 'kMean_' + slicenames[i] + slicenames_sub
        elif 'epsilonSGSmean' in properties and 'nuSGSmean' in properties:
            figname = 'epsilonMean_' + slicenames[i] + slicenames_sub
        else:
            figname = slicename

        if plot_type in ('2D', 'all'):
            slicePlot = Plot2D(x2d, y2d, vals3d, name=figname, xlabel=xlabel, ylabel=ylabel, val_label=val_label, save=save, show=show, figdir=case.result_path)
            slicePlot.initializeFigure()
            slicePlot.plotFigure(contour_lvl=contour_lvl)
            slicePlot.finalizeFigure()

    if plot_type in ('3D', 'all'):
        zlabel = r'$z$ [m]'
        # Figure name for 3D plots
        if 'kResolved' in properties and 'kSGSmean' in properties:
            figname_3d = 'kMean_' + str(slicenames)
        elif 'epsilonSGSmean' in properties and 'nuSGSmean' in properties:
            figname_3d = 'epsilonMean_' + str(slicenames)
        else:
            # Use raw slice name as figure name but remove spaces
            figname_3d = str(case.slicenames).replace(" ", '')

        if case.slices_orient[slicename] == 'horizontal':
            show_xylabel = (False, False)
            show_zlabel = True
            show_ticks = (False, False, True)
            if 'RGB' not in properties[0] and 'Rij' not in properties[0] and 'uuPrime2' not in properties[0]:
                multiplier = 1.75
                # Initialize plot object for horizontal contour slices
                plot3d = PlotContourSlices3D(list_x2d, list_y2d, list_val3d, horslice_offsets, gradient_bg=False, name=figname_3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, val_label=val_label[0], save=save, show=show, figdir=case.result_path, viewangle=view_angle, figwidth=figwidth, equalaxis=equalaxis, cbar_orient='vertical',
                                                  figheight_multiplier=multiplier,
                                                  val_lim=val_lim)
            else:
                multiplier = 2.25
                plot3d = PlotImageSlices3D(list_x2d, list_y2d, list_z2d, list_rgb=list_val3d,
                                           name=figname_3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                                           save=save, show=show, figdir=case.result_path, viewangle=view_angle, figwidth=figwidth,
                                           equalaxis=equalaxis, figheight_multiplier=multiplier)

            # If there's a z component e.g. Uz, initialize it separately
            if list_val3d_z[0] is not None:
                plot3d_z = PlotContourSlices3D(list_x2d, list_y2d, list_val3d_z, horslice_offsets, gradient_bg=False,
                                                  name=figname_3d + '_z', xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                                                  val_label=val_label[1], save=save, show=show, figdir=case.result_path,
                                                  viewangle=view_angle, figwidth=figwidth, equalaxis=equalaxis,
                                                  cbar_orient='vertical',
                                                  figheight_multiplier=1.75,
                                                  val_lim=val_lim_z,
                                               zlim=None)

        elif case.slices_orient[slicename] == 'vertical':
            show_xylabel = (True, True)
            show_zlabel = False
            show_ticks = (True, True, False)
            patch = Circle((0., 0.), 63., alpha=0.5, fill=False, edgecolor=(0.25, 0.25, 0.25), zorder=100)
            patches = []
            for i in range(100):
                patches.append(copy(patch))

            patches = iter(patches)
            # Initialize vertical surface plot instance
            plot3d = PlotSurfaceSlices3D(list_x2d, list_y2d, list_z2d, list_val3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, val_label=val_label[0], name=figname_3d, save=save, show=show, figdir=case.result_path, viewangle=view_angle, figwidth=figwidth, equalaxis=equalaxis, cbar_orient='horizontal',
                                              val_lim=val_lim)
            # Again separate instance for z component
            if list_val3d_z[0] is not None:
                plot3d_z = PlotSurfaceSlices3D(list_x2d, list_y2d, list_z2d, list_val3d_z, xlabel=xlabel, ylabel=ylabel,
                                                  zlabel=zlabel, val_label=val_label[1], name=figname_3d + '_z', save=save,
                                                  show=show, figdir=case.result_path, viewangle=view_angle,
                                                  figwidth=figwidth, equalaxis=equalaxis, cbar_orient='horizontal',
                                                  val_lim=val_lim_z)

        plot3d.initializeFigure(constrained_layout=True)
        plot3d.plotFigure(contour_lvl=contour_lvl)
        if casename not in ('ABL_N_H', 'ABL_N_L', 'ABL_N_L2', 'ABL_N_H_HiSpeed'):
            if case.slices_orient[slicename] == 'horizontal':
                for i in range(len(horslice_offsets)):
                    for j in range(len(turb_borders)):
                        plot3d.axes.plot([turb_borders[j][0], turb_borders[j][2]], [turb_borders[j][1], turb_borders[j][3]], zs=horslice_offsets2[i], alpha=0.5, color=(0.25, 0.25, 0.25),
                                         # Very important to set a super larger value
                                         zorder=500 + i*500)
            else:
                for i in range(len(list_x2d)):
                    p = next(patches)
                    plot3d.axes.add_patch(p)
                    pathpatch_2d_to_3d(p, z=0, normal=(0.8660254037844, 0.5, 0.))
                    pathpatch_translate(p, turb_centers_frontview[i])

        plot3d.finalizeFigure(tight_layout=False, show_ticks=show_ticks, show_xylabel=show_xylabel,
                              show_zlabel=False, z_ticklabel=(r'$z_\mathrm{hub}$', r'$z_\mathrm{mid}$', r'$z_\mathrm{apex}$'),
                              dpi=dpi)
        # For Uz or any other z component
        if list_val3d_z[0] is not None:
            plot3d_z.initializeFigure()
            plot3d_z.plotFigure(contour_lvl=contour_lvl)
            if casename not in ('ABL_N_H', 'ABL_N_L', 'ABL_N_L2', 'ABL_N_H_HiSpeed'):
                if case.slices_orient[slicename] == 'horizontal':
                    for i in range(len(horslice_offsets)):
                        for j in range(len(turb_borders)):
                            plot3d_z.axes.plot([turb_borders[j][0], turb_borders[j][2]], [turb_borders[j][1], turb_borders[j][3]],
                                             zs=horslice_offsets2[i], alpha=0.5, color=(0.25, 0.25, 0.25), zorder=500 + i*500)
                else:
                    for i in range(len(list_x2d)):
                        p = next(patches)
                        plot3d_z.axes.add_patch(p)
                        pathpatch_2d_to_3d(p, z=0, normal=(0.8660254037844, 0.5, 0.))
                        pathpatch_translate(p, turb_centers_frontview[i])

            plot3d_z.finalizeFigure(show_ticks=show_ticks, show_xylabel=show_xylabel,
                                    show_zlabel=False, z_ticklabel=(r'$z_\mathrm{hub}$', r'$z_\mathrm{mid}$', r'$z_\mathrm{apex}$'),
                                    dpi=dpi)








# """
# User Inputs
# """
# casedir = 'J:'  # '/media/yluan/Toshiba External Drive/'
# casedir = '/media/yluan/Toshiba External Drive/'
# casename = 'ALM_N_H_OneTurb'  # 'ALM_N_H_ParTurb'
# time = 23275.1388025  # 22000.0918025 20000.9038025
# # slicenames = ['alongWind', 'groundHeight', 'hubHeight', 'oneDaboveHubHeight', 'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# # For Upwind and Downwind turbines
# # slicenames = ['oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo', 'rotorPlaneOne', 'rotorPlaneTwo', 'sixDdownstreamTurbineTwo', 'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo', 'twoDupstreamTurbineOne']
# # # For Parallel Turbines
# # slicenames = ['alongWindRotorOne', 'alongWindRotorTwo', 'twoDupstreamTurbines', 'rotorPlane', 'oneDdownstreamTurbines', 'threeDdownstreamTurbines', 'sixDdownstreamTurbines']
# # slicenames = ['groundHeight', 'hubHeight', 'oneDaboveHubHeight']
# # slicenames = ['rotorPlane','sixDdownstreamTurbines']
# slicenames = ['alongWind']
# # Only for PlotContourSlices3D
# sliceOffsets = (5, 90, 153)
# propertyName = 'uuPrime2'
# fileExt = '.raw'
# precisionX, precisionY, precisionZ = 1000j, 1000j, 333j
# interp_method = 'nearest'
#
#
# """
# Plot Settings
# """
# figwidth = 'full'
# # View angle best (15, -40) for vertical slices in rotor plane
# view_angle, equalaxis = (15, -45), True
# xLim, yLim, zLim = (0, 3000), (0, 3000), (0, 1000)
# show, save = False, True
# xlabel, ylabel, zlabel = r'$x$ [m]', r'$y$ [m]', r'$z$ [m]'
# # valLabels = (r'$b_{11}$ [-]', r'$b_{12}$ [-]', r'$b_{13}$ [-]', r'$b_{22}$ [-]', r'$b_{23}$ [-]', r'$b_{33}$ [-]')
# # valLabels = (r'$\langle u\rangle$ [m/s]', r'$\langle v\rangle$ [m/s]', r'$\langle w\rangle$ [m/s]')
# if propertyName == 'U':
#     valLabels = (r'$U$ [m/s]', r'$U$ [m/s]', r'$U$ [m/s]')
# elif propertyName == 'uuPrime2':
#     valLabels = (r'$b_{11}$ [-]', r'$b_{12}$ [-]', r'$b_{13}$ [-]', r'$b_{22}$ [-]', r'$b_{23}$ [-]', r'$b_{33}$ [-]', r'$k_{\rm{resolved}}$ [m$^2$/s$^2$]')
#
#
# """
# Process User Inputs
# """
# # Combine propertyName with slicenames and Subscript to form the full file names
# # Don't know why I had to copy it...
# fileNames = slicenames.copy()
# for i, name in enumerate(slicenames):
#     slicenames[i] = propertyName + '_' + name + '_Slice'
#     fileNames[i] = slicenames[i] + fileExt
#
# figDir = casedir + casename + '/Slices/Result/' + str(time)
# try:
#     os.makedirs(figDir)
# except FileExistsError:
#     pass
#
#
# """
# Functions
# """
# @timer
# @jit
# def readSlices(time, casedir = '/media/yluan/Toshiba External Drive', casename = 'ALM_N_H', fileNames = ('*',), skipCol = 3, skipRow = 0):
#     caseFullPath = casedir + '/' + casename + '/Slices/' + str(time) + '/'
#     fileNames = os.listdir(caseFullPath) if fileNames[0] in ('*', 'all') else fileNames
#     slices_val, slicesDir, slices_coor = {}, {}, {}
#     for fileName in fileNames:
#         vals = np.genfromtxt(caseFullPath + fileName)
#         # partition('.') removes anything after '.'
#         slices_coor[fileName.partition('.')[0]] = vals[skipRow:, :skipCol]
#         # If max(z) - min(z) < 1 then it's assumed horizontal
#         slicesDir[fileName.partition('.')[0]] = 'vertical' if (vals[skipRow:, skipCol - 1]).max() - (vals[skipRow:, skipCol - 1]).min() > 1. else 'horizontal'
#         slices_val[fileName.partition('.')[0]] = vals[skipRow:, skipCol:]
#
#     print('\n' + str(fileNames) + ' read')
#     return slices_coor, slicesDir, slices_val
#
#
# @timer
# @jit
# def interpolateSlices(x, y, z, vals, sliceDir = 'vertical', precisionX = 1500j, precisionY = 1500j, precisionZ = 500j, interp_method = 'cubic'):
#     # Bound the coordinates to be interpolated in case data wasn't available in those borders
#     bnd = (1.00001, 0.99999)
#     if sliceDir is 'vertical':
#         # Known x and z coordinates, to be interpolated later
#         knownPoints = np.vstack((x, z)).T
#         # Interpolate x and z according to precisions
#         x2d, z2d = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
#         # Then interpolate y in the same fashion of x
#         y2d, _ = np.mgrid[y.min()*bnd[0]:y.max()*bnd[1]:precisionY, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
#         # In case the vertical slice is at a negative angle,
#         # i.e. when x goes from low to high, y goes from high to low,
#         # flip y2d from low to high to high to low
#         y2d = np.flipud(y2d) if x[0] > x[1] else y2d
#     else:
#         knownPoints = np.vstack((x, y)).T
#         x2d, y2d = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, y.min()*bnd[0]:y.max()*bnd[1]:precisionY]
#         _, z2d = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX, z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
#
#     # Decompose the vector/tensor of slice values
#     # If vector, order is x, y, z
#     # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
#     valsDecomp = {}
#     for i in range(vals.shape[1]):
#         if sliceDir is 'vertical':
#             # Each component is interpolated from the known locations pointsXZ to refined fields (x2d, z2d)
#             valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2d, z2d), method = interp_method)
#         else:
#             valsDecomp[str(i)] = griddata(knownPoints, vals[:, i].ravel(), (x2d, y2d), method = interp_method)
#
#     return x2d, y2d, z2d, valsDecomp
#
#
# @timer
# @jit
# def calculateAnisotropicTensor(valsDecomp):
#     # k in the interpolated mesh
#     # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
#     k = 0.5*(valsDecomp['0'] + valsDecomp['3'] + valsDecomp['5'])
#     # Convert Rij to bij
#     for key, val in valsDecomp.items():
#         valsDecomp[key] = val/(2.*k) - 1/3. if key in ('0', '3', '5') else val/(2.*k)
#
#     return valsDecomp, k
#
#
# @timer
# @jit
# def mergeHorizontalComponent(valsDecomp):
#     valsDecomp['hor'] = np.sqrt(valsDecomp['0']**2 + valsDecomp['1']**2)
#     return valsDecomp
#
#
# """
# Read, Decompose and Plot 2/3D Slices
# """
# slices_coor, slicesDir, slices_val = readSlices(time = time, casedir = casedir, casename = casename, fileNames = fileNames)
#
# # Initialize slice lists for multple slice plots in one 3D figure
# horSliceLst, zSliceLst, list_x2d, list_y2d, list_z2d = [], [], [], [], []
# # Go through slices
# for slicename in slicenames:
#     x2d, y2d, z2d, valsDecomp = interpolateSlices(slices_coor[slicename][:, 0], slices_coor[slicename][:, 1], slices_coor[slicename][:, 2], slices_val[slicename], sliceDir = slicesDir[slicename], precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interp_method = interp_method)
#
#     # For anisotropic stress tensor bij
#     # bij = Rij/(2k) - 1/3*deltaij
#     # where Rij is uuPrime2, k = 1/2trace(Rij), deltaij is Kronecker delta
#     if propertyName == 'uuPrime2':
#         valsDecomp, k = calculateAnisotropicTensor(valsDecomp)
#         valsDecomp['kResolved'] = k
#     elif propertyName == 'U':
#         valsDecomp = mergeHorizontalComponent(valsDecomp)
#
#
#     """
#     2D Contourf Plots
#     """
#     xLim, yLim, zLim = (x2d.min(), x2d.max()), (y2d.min(), y2d.max()), (z2d.min(), z2d.max())
#     plotsLabel = iter(valLabels)
#     for key, val in valsDecomp.items():
#         # if slicesDir[slicename] is 'vertical':
#         #     slicePlot = Plot2D(x2d, z2d, z2d = val, equalaxis = True,
#         #                                  name = slicename + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim,
#         #                                  show = show, xlabel = xlabel, ylabel = zlabel, save = save,
#         #                                  zlabel = next(plotsLabel))
#         #
#         # else:
#         #     slicePlot = Plot2D(x2d, y2d, z2d = val, equalaxis = True,
#         #                        name = slicename + '_' + key, figDir = figDir, xLim = xLim, yLim = yLim,
#         #                        show = show, xlabel = xlabel, ylabel = ylabel, save = save,
#         #                        zlabel = next(plotsLabel))
#         # slicePlot = Plot2D_InsetZoom(x2d, z2d, zoomBox = (1000, 2500, 0, 500), z2d = val, equalaxis = True, name = slicename + '_' + key, figDir = figDir, xLim = xLim, yLim = zLim, show = show, xlabel = xlabel, ylabel = zlabel, save = save, zlabel = next(plotsLabel))
#         # plot_type = 'contour2D'
#
#         slicePlot = PlotSurfaceSlices3D(x2d, y2d, z2d, val, name = slicename + '_' + key + '_3d', figDir = figDir, xLim = xLim, yLim = yLim, zLim = zLim, show = show, xlabel = xlabel, ylabel = ylabel, zlabel = zlabel, save = save, cmapLabel = next(plotsLabel), viewAngles = view_angle, figwidth = figwidth)
#         plot_type = 'surface3D'
#
#         slicePlot.initializeFigure()
#         if plot_type == 'contour2D':
#             slicePlot.plotFigure(contour_lvl = 100)
#         else:
#             slicePlot.plotFigure()
#
#         slicePlot.finalizeFigure()
#
#     if propertyName == 'U':
#         horSliceLst.append(valsDecomp['hor'])
#         zSliceLst.append(valsDecomp['2'])
#
#     list_x2d.append(x2d)
#     list_y2d.append(y2d)
#     list_z2d.append(z2d)


"""
Multiple Slices of Horizontal Component 3D Plot
"""
# if slicesDir[slicename] is 'horizontal':
#     slicePlot = PlotContourSlices3D(x2d, y2d, horSliceLst, sliceOffsets = sliceOffsets, contour_lvl = 100, zLim = (0, 216), gradientBg = False, name = str(slicenames) + '_hor', figDir = figDir, show = show, xlabel = xlabel, ylabel = ylabel, zlabel = zlabel, cmapLabel = r'$U_{\rm{hor}}$ [m/s]', save = save, cbarOrientate = 'vertical')
# else:
#     slicePlot = PlotSurfaceSlices3D(list_x2d, list_y2d, list_z2d, horSliceLst, name = str(slicenames) + '_hor', figDir = figDir, show = show, xlabel = xlabel,
#                                     ylabel = ylabel, zlabel = zlabel, save = save, cmapLabel = r'$U_{\rm{hor}}$ [m/s]', viewAngles = view_angle, figwidth = figwidth, xLim = xLim, yLim = yLim, zLim = zLim, equalaxis = equalaxis)
#
# slicePlot.initializeFigure()
# slicePlot.plotFigure()
# slicePlot.finalizeFigure()


"""
Multiple Slices of Z Component 3D Plot
"""
# if slicesDir[slicename] is 'horizontal':
#     slicePlot = PlotContourSlices3D(x2d, y2d, zSliceLst, sliceOffsets = sliceOffsets, contour_lvl = 100,
#                                     xLim = (0, 3000), yLim = (0, 3000), zLim = (0, 216), gradientBg = False,
#                                     name = str(slicenames) + '_z', figDir = figDir, show = show,
#                                     xlabel = xlabel, ylabel = ylabel, zlabel = zlabel,
#                                     cmapLabel = r'$U_{z}$ [m/s]', save = save, cbarOrientate = 'vertical')
# else:
#     slicePlot = PlotSurfaceSlices3D(list_x2d, list_y2d, list_z2d, zSliceLst,
#                                     name = str(slicenames) + '_z', figDir = figDir, show = show, xlabel = xlabel,
#                                     ylabel = ylabel, zlabel = zlabel, save = save, cmapLabel = r'$U_{z}$ [m/s]', viewAngles = view_angle, figwidth = figwidth, xLim = xLim, yLim = yLim, zLim = zLim, equalaxis = equalaxis)
#
# slicePlot.initializeFigure()
# slicePlot.plotFigure()
# slicePlot.finalizeFigure()




