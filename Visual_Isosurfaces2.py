#!/usr/bin/python
import numpy as np
from FieldData import FieldData
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
casedir = 'J:'
# casedir = '/media/yluan/Toshiba External Drive/'
casename = 'ALM_N_H_ParTurb'
times = '22000.0918025'
fields = 'Q'
# Target number of cells after interpolation
targetRes = 1000000  # 'min', <int>
# # Not used
# precisionX, precisionY, precisionZ = 500j, 500j, 167j
# Not used
interpMethod = 'linear'
pickleName = 'turb0'
# Force to recalculate values even if pickle results exist?
force_recalc = False  # [CAUTION]

# Initialize the case
case = FieldData(fields=fields, times=times, casename=casename, casedir=casedir)
# Go through all specified time directories
for time in case.times:
    # Check if pickle results saved for this pickleName and fields
    # If so, use pickle results
    resultnames = os.listdir(case.result_paths[time])
    use_pickle = True if pickleName + '_' + fields + '3D.p' in resultnames else False
    # If no pickle results stored, then run the whole process
    if not use_pickle or force_recalc:
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
        if casename == 'ALM_N_H_ParTurb':
            # For northern turbine a.k.a. turb1 in ALM_N_H_ParTurb
            if pickleName == 'turb1':
                # Origin
                boxO = (914.464, 1380.179 - 2.5, 0)
            # For southern turbine a.k.a. turb0 in ALM_N_H_ParTurb
            elif pickleName == 'turb0':
                boxO = (1103.464, 1052.821 - 2.5, 0)

            boxL, boxW, boxH = 6*126, 63 + 2.5*2, 216
        elif casename == 'ALM_N_H':
            boxO= (1008.964, 1216.5 - 2.5, 0)
            boxL, boxW, boxH = 1638, 63 + 2.5*2, 216

        boxLWH = boxL + boxW + boxH
        ar = (boxL/boxLWH, boxW/boxLWH, boxH/boxLWH)
        targetRes = len(ccx)*1.1 if targetRes == 'min' else targetRes
        avgPrecision = (targetRes/ar[0]/ar[1]/ar[2])**(1/3.)
        precisionX, precisionY, precisionZ = np.int(np.ceil(ar[0]*avgPrecision))*1j,\
                                             np.int(np.ceil(ar[1]*avgPrecision))*1j, \
                                             np.int(np.ceil(ar[2]*avgPrecision))*1j
        print(precisionX, precisionY, precisionZ)
        # Confine to domain of interest
        ccx, ccy, ccz, data, box, flags = case.confineFieldDomain_Rotated(ccx, ccy, ccz, data, boxL = boxL,
                                                                          boxW = boxW, boxH = boxH,
                                                                          boxO = boxO, boxRot = boxRot)

        ccx3D, ccy3D, ccz3D, data3D = case.interpolateFieldData(ccx, ccy, ccz, data, precisionX = precisionX, precisionY = precisionY, precisionZ = precisionZ, interpMethod = interpMethod)
        # ccx3D, ccy3D, ccz3D, data3D = case.interpolateFieldData_RBF(ccx, ccy, ccz, data, precisionX = precisionX,
        #                                                         precisionY = precisionY, precisionZ = precisionZ,
        #                                                         function = interpMethod)

        print('\nDumping results')
        pickle.dump(ccx3D, open(case.result_path[time] + pickleName + '_ccx3D.p', 'wb'))
        pickle.dump(ccy3D, open(case.result_path[time] + pickleName + '_ccy3D.p', 'wb'))
        pickle.dump(ccz3D, open(case.result_path[time] + pickleName + '_ccz3D.p', 'wb'))
        pickle.dump(data3D, open(case.result_path[time] + pickleName + '_' + fields + '3D.p', 'wb'))


        """
        Mayavi Quiver Visualization
        """
        # Start engine, don't know why
        engine = Engine()
        engine.start()
        axes = Axes()
        mlab.figure(pickleName + '_' + fields, engine = engine, bgcolor = (1, 1, 1),
                    fgcolor = (0.5, 0.5, 0.5))
        mlab.contour3d(ccx3D, ccy3D, ccz3D, data3D, contours = [1.], color = (244/255., 66/255., 66/255.))


