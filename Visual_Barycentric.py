import numpy as np
from PostProcess_FieldData import FieldData
import PostProcess_AnisotropyTensor as ppat
import time as t
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mayavi import mlab
from mayavi.api import Engine
import pickle

caseDir = 'J:'
caseName = 'ALM_N_H'
times = '22000.0918025'
fields = 'uuPrime2'
# Not used
precisionX, precisionY, precisionZ = 100j, 100j, 33j
interpMethod = 'linear'
pickleName = 'turbs'
usePickle = False


case = FieldData(fields = fields, times = times, caseName = caseName, caseDir = caseDir)


if not usePickle:
    # [BUG]
    # The keyword for symmetric tensor in parse_data_nonuniform() of field_parser.py of Ofpp should be 'symmTensor'
    # instead of 'symmtensor'
    fieldData = case.readFieldData()

    data = fieldData[fields]

    ccx, ccy, ccz, cc = case.readCellCenterCoordinates()

    if caseName is 'ALM_N_H_ParTurb':
        # For northern turbine a.k.a. turb1 in ALM_N_H_ParTurb
        if pickleName is 'turb1':
            boxO = (914.464, 1380.179 - 2.5, 0)
        # For southern turbine a.k.a. turb0 in ALM_N_H_ParTurb
        elif pickleName is 'turb0':
            boxO = (1103.464, 1052.821 - 2.5, 0)

        boxL, boxW, boxH = 6*126, 63 + 2.5*2, 216
    elif caseName is 'ALM_N_H':
        boxO = (1008.964, 1216.5 - 2.5, 0)
        boxL, boxW, boxH = 1638, 63 + 2.5*2, 216

    ccx, ccy, ccz, data, box, flags = case.confineFieldDomain_Rotated(ccx, ccy, ccz, data, boxL = 6*126, boxW = 63 + 2.5*2, boxH = 216, boxO = boxO, boxRot = np.pi/6.)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # patch = patches.PathPatch(box, facecolor='orange', lw=0)
    # ax.add_patch(patch)
    # ax.set_xlim(0,3000)
    # ax.set_ylim(0,3000)
    # plt.show()

    # Process anisotropy tensors
    if fields is 'uuPrime2':
        t0 = t.time()
        data, tensors, eigVals3D, eigVecs4D = ppat.processAnisotropyTensor_Uninterpolated(data)
        t1 = t.time()
        ticToc = t1 - t0

        print('\nDumping results')
        for time in case.times:
            # Field data
            pickle.dump(ccx, open(case.resultPath[time] + pickleName + '_ccx.p', 'wb'))
            pickle.dump(ccx, open(case.resultPath[time] + pickleName + '_ccy.p', 'wb'))
            pickle.dump(ccx, open(case.resultPath[time] + pickleName + '_ccz.p', 'wb'))
            pickle.dump(data, open(case.resultPath[time] + pickleName + '_' + fields + '.p', 'wb'))
            pickle.dump(tensors, open(case.resultPath[time] + pickleName + '_' + fields + '_tensors.p', 'wb'))
            pickle.dump(eigVals3D, open(case.resultPath[time] + pickleName + '_' + fields + '_eigVals.p', 'wb'))
            pickle.dump(eigVecs4D, open(case.resultPath[time] + pickleName + '_' + fields + '_eigVecs.p', 'wb'))

        # x3D, y3D, z3D, data = case.interpolateFieldData_RBF(ccx, ccy, ccz, data, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, function = interpMethod)

else:
    for time in case.times:
        ccx = pickle.load(open(case.resultPath[time] + pickleName + '_ccx.p', 'rb'), encoding = 'latin1')
        ccy = pickle.load(open(case.resultPath[time] + pickleName + '_ccy.p', 'rb'), encoding = 'latin1')
        ccz = pickle.load(open(case.resultPath[time] + pickleName + '_ccz.p', 'rb'), encoding = 'latin1')
        eigVals3D = pickle.load(open(case.resultPath[time] + pickleName + '_' + fields + '_eigVals.p', 'rb'), encoding = 'latin1')
        eigVecs4D = pickle.load(open(case.resultPath[time] + pickleName + '_' + fields + '_eigVecs.p', 'rb'), encoding = 'latin1')


# Eigenvector manipulation, reverse it by -1
eigVecs4D *= -1.

"""
Mayavi Quiver Visualization
"""
engine = Engine()
engine.start()
# if len(engine.scenes) == 0:
#     engine.new_scene()

mlab.figure(pickleName + '_quivers', engine = engine, size = (1000, 800), bgcolor = (1, 1, 1), fgcolor = (0.5, 0.5, 0.5))
quiver = mlab.quiver3d(ccx, ccy, ccz, eigVecs4D[:, :, 0, 0].ravel(), eigVecs4D[:, :, 0, 1].ravel(), eigVecs4D[:, :, 0, 2].ravel(), scalars = eigVals3D[:, :, 0].ravel(), mask_points = 100, scale_mode = 'scalar', colormap = 'plasma')
mlab.outline()
quiver.glyph.color_mode = 'color_by_scalar'

# src = mlab.pipeline.vector_field(ccx, ccy, ccz, eigVecs4D[:, :, 0, 0].ravel(), eigVecs4D[:, :, 0, 1].ravel(), eigVecs4D[:, :, 0, 2].ravel())
# mlab.pipeline.vectors(src, mask_points = 20, scale_factor = 3.)


# meshSize, cellSizeMin, ccx3D, ccy3D, ccz3D = case.getMeshInfo(ccx, ccy, ccz)


