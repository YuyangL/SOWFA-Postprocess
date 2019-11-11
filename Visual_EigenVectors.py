import numpy as np
from FieldData import FieldData
import PostProcess_AnisotropyTensor as ppat
import time as t
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mayavi import mlab
from mayavi.api import Engine
from mayavi.modules.axes import Axes
import pickle
import os

"""
User Inputs
"""
caseDir = 'J:'
caseDir = '/media/yluan/1'
caseName = 'ALM_N_H'
times = '22000.0918025'
fields = 'uuPrime2'
# # Not used
# precisionX, precisionY, precisionZ = 100j, 100j, 33j
# # Not used
# interpMethod = 'linear'
pickleName = 'turbs'
figView = 'top'  # 'iso', 'front', 'left', 'top'


"""
Process u'u' Tensor
"""
# Initialize the case
case = FieldData(fields = fields, times = times, caseName = caseName, caseDir = caseDir)
# Go through all specified time directories
for time in case.times:
    # Check if pickle results saved for this pickleName and fields
    # If so, use pickle results
    resultNames = os.listdir(case.resultPath[time])
    usePickle = True if pickleName + '_' + fields + '.p' in resultNames else False
    # If no pickle results stored, then run the whole process
    if not usePickle:
        # [BUG]
        # The keyword for symmetric tensor in parse_data_nonuniform() of field_parser.py of Ofpp should be 'symmTensor'
        # instead of 'symmtensor'
        fieldData = case.readFieldData()
        # The data
        data = fieldData[fields]
        # Coordinates of the whole domain in 1D arrays
        ccx, ccy, ccz, cc = case.readCellCenterCoordinates()
        # Confine the domain of interest
        # For paralllel turbines, the confine box starts from the center of each rotor plane,
        # 1D upstream turbines, length is 6D,  width is 0.5D, height is 1D above hub height
        # For sequential turbines, the confine box starts from the center of rotor planes,
        # 1D upstream upwind turbine, length is 13D, width is 0.5D, height is 1D above hub height
        # Box counter-clockwise rotation in x-y plane
        boxRot = np.pi/6.
        if caseName == 'ALM_N_H_ParTurb':
            # For northern turbine a.k.a. turb1 in ALM_N_H_ParTurb
            if pickleName == 'turb1':
                # Origin
                boxO = (914.464, 1380.179 - 2.5, 0)
            # For southern turbine a.k.a. turb0 in ALM_N_H_ParTurb
            elif pickleName == 'turb0':
                boxO = (1103.464, 1052.821 - 2.5, 0)

            boxL, boxW, boxH = 6*126, 63 + 2.5*2, 216
        elif caseName == 'ALM_N_H':
            boxO= (1008.964, 1216.5 - 2.5, 0)
            boxL, boxW, boxH = 1638, 63 + 2.5*2, 216

        # Confine to domain of interest
        ccx, ccy, ccz, data, box, flags = case.confineFieldDomain_Rotated(ccx, ccy, ccz, data, boxL = 6*126, boxW = 63 + 2.5*2, boxH = 216, boxO = boxO, boxRot = boxRot)

        # # Visualize the confine box
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # patch = patches.PathPatch(box, facecolor='orange', lw=0)
        # ax.add_patch(patch)
        # ax.set_xlim(0,3000)
        # ax.set_ylim(0,3000)
        # plt.show()

        # Process anisotropy tensors
        t0 = t.time()
        data, tensors, eigVals3D, eigVecs4D = ppat.processAnisotropyTensor_Uninterpolated(data)
        t1 = t.time()
        ticToc = t1 - t0

        print('\nDumping results')
        pickle.dump(ccx, open(case.resultPath[time] + pickleName + '_ccx.p', 'wb'))
        pickle.dump(ccy, open(case.resultPath[time] + pickleName + '_ccy.p', 'wb'))
        pickle.dump(ccz, open(case.resultPath[time] + pickleName + '_ccz.p', 'wb'))
        pickle.dump(data, open(case.resultPath[time] + pickleName + '_' + fields + '.p', 'wb'))
        pickle.dump(tensors, open(case.resultPath[time] + pickleName + '_' + fields + '_tensors.p', 'wb'))
        pickle.dump(eigVals3D, open(case.resultPath[time] + pickleName + '_' + fields + '_eigVals.p', 'wb'))
        pickle.dump(eigVecs4D, open(case.resultPath[time] + pickleName + '_' + fields + '_eigVecs.p', 'wb'))

        # # Not used, interpolation
        # x3D, y3D, z3D, data = case.interpolateFieldData_RBF(ccx, ccy, ccz, data, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, function = interpMethod)

    # If existing pickle results found for pickleName, load them
    else:
        print('\nExisting pickle results found for ' + pickleName + ', loading them...')
        ccx = pickle.load(open(case.resultPath[time] + pickleName + '_ccx.p', 'rb'), encoding = 'latin1')
        ccy = pickle.load(open(case.resultPath[time] + pickleName + '_ccy.p', 'rb'), encoding = 'latin1')
        ccz = pickle.load(open(case.resultPath[time] + pickleName + '_ccz.p', 'rb'), encoding = 'latin1')
        eigVals3D = pickle.load(open(case.resultPath[time] + pickleName + '_' + fields + '_eigVals.p', 'rb'), encoding = 'latin1')
        eigVecs4D = pickle.load(open(case.resultPath[time] + pickleName + '_' + fields + '_eigVecs.p', 'rb'), encoding = 'latin1')


    """
    Mayavi Quiver Visualization
    """
    # Start engine, don't know why
    engine = Engine()
    engine.start()
    axes = Axes()
    mlab.figure(pickleName + '_quivers', engine = engine, size = (1000, 800), bgcolor = (1, 1, 1), fgcolor = (0.5, 0.5, 0.5))
    quiver = mlab.quiver3d(ccx, ccy, ccz, eigVecs4D[:, :, 0, 0].ravel(), eigVecs4D[:, :, 0, 1].ravel(), eigVecs4D[:, :, 0, 2].ravel(), scalars = eigVals3D[:, :, 0].ravel(), mask_points = 150, scale_mode = 'scalar', colormap = 'plasma', opacity = 1)
    # mlab.outline()
    # Added axis
    engine.add_filter(axes, quiver)
    quiver.glyph.color_mode = 'color_by_scalar'
    quiver.glyph.glyph_source.glyph_source.glyph_type = 'dash'
    scene = engine.scenes[0]
    scene.scene.jpeg_quality = 100
    scene.scene.anti_aliasing_frames = 20
    # Axis related settings
    axes.axes.x_label = 'x [m]'
    axes.axes.y_label = 'y [m]'
    axes.axes.z_label = 'z [m]'
    axes.title_text_property.bold, axes.label_text_property.bold = False, False
    axes.label_text_property.italic = False
    axes.title_text_property.font_family = 'times'
    # Axis texts scales to fit in the viewport?
    axes.axes.scaling = False
    # Text color
    axes.title_text_property.color, axes.label_text_property.color = (0, 0, 0), (89/255., 89/255., 89/255.)
    # Text size
    axes.title_text_property.font_size, axes.label_text_property.font_size = 14, 12
    axes.axes.font_factor = 1.0
    # Prevent corner axis label clash
    axes.axes.corner_offset = 0.05
    figW = 3.39
    figH = figW*(np.sqrt(5) - 1.0)/2.0
    if figView is 'iso':
        mlab.view(azimuth = 260, elevation = 60)
        # Move the figure left 20 pixels?
        mlab.move(right = -20)
    elif figView is 'front':
        mlab.view(azimuth = 210, elevation = 90)
        mlab.move(forward = 500)
    elif figView is 'left':
        mlab.view(azimuth = 120, elevation = 90)
        mlab.move(up = 0, right = -40)
    elif figView is 'top':
        mlab.view(azimuth = 0, elevation = 0)
        mlab.move(right = -20)

    # [BUG] Magnification doesn't work on axis
    mlab.savefig(case.resultPath[time] + pickleName + '_quiver_' + figView + '.png', size = (figW, figH))
    # mlab.show()



