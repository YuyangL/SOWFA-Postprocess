import os
import numpy as np
from warnings import warn
import Ofpp as of
import random
from numba import njit, prange
from Utility import timer, deprecated
from matplotlib import path
from scipy.interpolate import griddata, Rbf
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

class FieldData:
    def __init__(self, fields='*', times='latestTime', casename='ABL_N_H', casedir='.', filenames_pre='', filenames_sub='',
                 cc=('ccx', 'ccy', 'ccz'), result_folder='Result', field_foldername='Fields', save=True, save_protocol=4):
        self.fields = fields
        self.case_fullpath = casedir + '/' + casename + '/' + field_foldername + '/'
        self.filenames_pre, self.filenames_sub = filenames_pre, filenames_sub
        self.cc = cc
        self.case_time_fullpaths, self.result_paths = {}, {}
        # Save result as pickle and based on protocol pyVer
        self.save, self.save_protocol = save, save_protocol
        # If times in list/tuple or all times requested
        if isinstance(times, (list, tuple)) or times in ('*', 'all', 'All', 'latestTime'):
            # If all times, try remove the result folder from found directories
            if times in ('*', 'all', 'All', 'latestTime'): self.times = self._readTimes(removes=result_folder)[1]
            if times == 'latestTime' and not isinstance(self.times, str): self.times = self.times[-1]
            if isinstance(self.times, str): self.times = [self.times]
            # Go through all provided times, time could be string or integer and/or float
            for time in self.times:
                self.case_time_fullpaths[str(time)] = self.case_fullpath + str(time) + '/'
                # In the result directory, there are time directories
                self.result_paths[str(time)] = casedir + '/' + casename + '/' + field_foldername + '/' + result_folder + '/' + str(time) + '/'
                # Try to make the result directories, if not existent already
                os.makedirs(self.result_paths[str(time)], exist_ok=True)

        # Else if only one time provided, time could be string or integer or float
        else:
            self.case_time_fullpaths[str(times)] = self.case_fullpath + str(times) + '/'
            # Make sure times is in list(str)
            self.times = [str(times)]
            self.result_paths[str(times)] = casedir + '/' + casename + '/Fields/' + result_folder + '/' + str(times) + '/'
            os.makedirs(self.result_paths[str(times)], exist_ok=True)

        # If fields is 'all'/'All'/'*'
        # Get a random time directory in order to collect fields
        _, self.case_time_fullpaths_rand = random.choice(list(self.case_time_fullpaths.items()))
        if fields in ('all', 'All', '*'):
            self.fields = os.listdir(self.case_time_fullpaths_rand)
            # Remove cellCenter fields from this list
            self.fields = [val for _, val in enumerate(self.fields) if val not in cc]
            # Try removing uniform folder if it exists
            try:
                self.fields.remove('uniform')
            except OSError:
                pass

        # Else if provided fields not in a list/tuple
        # Convert str to list
        elif not isinstance(fields, (list, tuple)):
            self.fields = [fields]

        # Indices
        self.ij_uniq = [0, 1, 2, 4, 5, 8]
        self.ii6, self.ii9 = [0, 3, 5], [0, 4, 8]
        self.ij_6to9 = [0, 1, 2, 1, 3, 4, 2, 4, 5]
        print('\nFieldData object initialized')

    def _readTimes(self, removes=('Result',)):
        times_all = os.listdir(self.case_fullpath)
        # Ensure tuple so it can be looped
        if isinstance(removes, str): removes = (removes,)
        # Remove strings on request
        for remove in removes:
            try:
                times_all.remove(remove)
            except ValueError:
                pass

        # Raw all times that are string and can be integer and float mixed
        # Useful for locating time directories that can be integer
        times_all_raw = times_all
        # Numerical float all times and sort from low to high
        times_all = np.array([float(i) for i in times_all])
        # Sort string all times by its float counterpart
        times_all_raw = [time_raw for time, time_raw in sorted(zip(times_all, times_all_raw))]
        # Use Numpy sort() to sort float all times
        times_all.sort()

        return times_all, times_all_raw

    @timer
    def readFieldData(self):
        """
        Read field data specified by self.fields and self.times, data shape is nPoint (x nComponent (x nTime)).
        E.g. scalar field with one time is nPoint x 0 x 0;
        tensor field with one time is nPoint x nComponent x 0;
        :return: field_data dictionary
        :rtype: dict(np.ndarray(nPoint (x nComponent (x nTime))))
        """
        field_data = {}
        # Go through each specified field
        for i in range(len(self.fields)):
            field = self.fields[i]
            print(' Reading {}...'.format(field))
            # Read the data of field in the 1st time directory
            field_data[field] = of.parse_internal_field(self.case_time_fullpaths[self.times[0]] + field)
            # If multiple times requested, read data and stack them in 3rd D
            if len(self.times) > 1:
                for j in range(1, len(self.times)):
                    field_data_j = of.parse_internal_field(self.case_time_fullpaths[self.times[j]] + field)
                    field_data[field] = np.dstack((field_data[field], field_data_j))

                # When multiple times, since a 3rd dimension is added and Numpy.dstack treats scalar fields of shape (nPoint,) as (1, nPoint),
                # reshape scalar fields of shape (1, nPoint, nTime) back to (nPoint, 1, nTime)
                field_data[field] = field_data[field].reshape((
                    field_data[field].shape[1],
                    field_data[field].shape[0],
                    field_data[field].shape[2]
                )) if field_data[field].shape[0] == 1 else field_data[field]

        print('\n{0} data read for {1} s. \nIf multiple times requested, data of different times are stacked in 3D'.format(self.fields, self.times))
        return field_data

    @staticmethod
    @timer
    @deprecated
    def rotateSpatialCorrelationTensors(list_data, rotateXY=0., rotateUnit='rad', dependencies=('xx',)):
        # FIXME: DEPRECATED
        """
        Rotate one or more single/double spatial correlation scalar/tensor field/slice data in the x-y plane,
        doesn't work on rate of strain/rotation tensors
        :param list_data: Any of (nPoint x nComponent) or (nX x nY x nComponent) or (nX x nY x nZ x nComponent) data of interest, appended to a tuple/list.
        If nComponent is 6, data is symmetric 3 x 3 double spatial correlation tensor field.
        If nComponent is 9, data is single/double spatial correlation tensor field depending on dependencies keyword
        :type list_data: tuple/list([:, :]/[:, :, :]/[:, :, :, :])
        :param rotateXY:
        :type rotateXY:
        :param rotateUnit:
        :type rotateUnit:
        :param dependencies: Only used if nComponent is 9.
        Whether the component of data is dependent on single spatial correlation 'x' e.g. gradient, vector;
        or double spatial correlation 'xx' e.g. double correlation uu, d^2u/dx^2
        :type dependencies: str or list/tuple of 'x' or 'xx'. Default is ('xx',)
        :return: listData_rot
        :rtype:
        """
        # Ensure list input since each list_data[i] is modified to nPt x nComponent later
        list_data = list((list_data,)) if isinstance(list_data, np.ndarray) else list(list_data)
        # Ensure tuple input
        dependencies = (dependencies,) if isinstance(dependencies, str) else dependencies
        # Ensure dependencies has the same entries as the number of data provided
        dependencies *= len(list_data) if len(dependencies) < len(list_data) else 1
        # Ensure radian unit
        rotateXY *= np.pi/180 if rotateUnit != 'rad' else 1.
        # Reshape here screwed up njit :/
        def __transform(list_data, rotateXY, dependencies):
            # Copy list_data (a list) that has original shapes as list_data will be flattened to nPt x nComponent
            listDataRot_oldShapes = list_data.copy()
            sinVal, cosVal = np.sin(rotateXY), np.cos(rotateXY)
            # Go through every data in list_data and flatten to nPt x nComponent if necessary
            for i in range(len(list_data)):
                # # Ensure Numpy array
                # list_data[i] = np.array(list_data[i])
                # Flatten data from 3D to 2D
                if len(list_data[i].shape) >= 3:
                    # Go through 2nd D to (last - 1) D
                    nRow = 1
                    for j in range(len(list_data[i].shape) - 1):
                        nRow *= list_data[i].shape[j]

                    # Flatten data to nPt x nComponent
                    nComponent = list_data[i].shape[len(list_data[i].shape) - 1]
                    list_data[i] = list_data[i].reshape((nRow, nComponent))

            # Create a copy so list_data values remain unchanged during the transformation, list_data[i] is nPt x nComponent now
            listData_rot = list_data.copy()
            # Go through all provided data and perform transformation
            for i in range(len(list_data)):
                # Number of component is the last D
                nComponent = list_data[i].shape[len(list_data[i].shape) - 1]
                # If nComponent is 3, i.e. x, y, z or data is single spatial correlation with 9 components
                if nComponent == 3 or (nComponent == 9 and dependencies[i] == 'x'):
                    # x_rot becomes x*cos + y*sin
                    x_rot = list_data[i][:, 0]*cosVal + list_data[i][:,
                                                       1]*sinVal
                    # y_rot becomes -x*sin + y*cos
                    y_rot = -list_data[i][:, 0]*sinVal + list_data[i][:,
                                                        1]*cosVal
                    # z_rot doesn't change
                    z_rot = list_data[i][:, 2]
                    listData_rot[i][:, 0], listData_rot[i][:, 1], listData_rot[i][:, 2] = x_rot, y_rot, z_rot
                    # If 9 components with single spatial correlation, e.g.gradient tensor, do it for 2nd row and 3rd row
                    if nComponent == 9:
                        x_rot2 = list_data[i][:, 3]*cosVal + list_data[i][:,
                                                            4]*sinVal
                        y_rot2 = -list_data[i][:, 3]*sinVal + list_data[i][:,
                                                             4]*cosVal
                        z_rot2 = list_data[i][:, 5]
                        x_rot3 = list_data[i][:, 6]*cosVal + list_data[i][:,
                                                            7]*sinVal
                        y_rot3 = -list_data[i][:, 6]*sinVal + list_data[i][:,
                                                             7]*cosVal
                        z_rot3 = list_data[i][:, 8]
                        listData_rot[i][:, 3], listData_rot[i][:, 4], listData_rot[i][:, 5] = x_rot2, y_rot2, z_rot2
                        listData_rot[i][:, 6], listData_rot[i][:, 7], listData_rot[i][:, 8] = x_rot3, y_rot3, z_rot3

                # Else if nComponent is 6 or 9 with double spatial correlation
                elif nComponent == 6 or (nComponent == 9 and dependencies[i] == 'xx'):
                    # If 6 components and double spatial correlation, i.e.xx, xy, xz, yy, yz, zz
                    # or 9 components and double spatial correlation, i.e. xx, xy, xz, yx, yy, yz, zx, zy, zz
                    if dependencies[i] == 'xx':
                        xx, xy, xz = list_data[i][:, 0], list_data[i][:, 1], list_data[i][:, 2]
                        yy, yz = list_data[i][:, 3], list_data[i][:, 4]
                        zz = list_data[i][:, 5]
                        # xx_rot becomes x_rot*x_rot = xx*cos^2 + 2xy*sin*cos + yy*sin^2
                        xx_rot = xx*cosVal**2 + 2*xy*sinVal*cosVal + yy*sinVal**2
                        # xy_rot becomes x_rot*y_rot = xy*cos^2 + yy*sin*cos - xx*sin*cos -xy*sin^2
                        xy_rot = xy*cosVal**2 + yy*sinVal*cosVal - xx*sinVal*cosVal - xy*sinVal**2
                        # xz_rot become x_rot*z_rot = xz*cos + yz*sin
                        xz_rot = xz*cosVal + yz*sinVal
                        # yy_rot becomes y_rot*y_rot = yy*cos^2 - 2xy*sin*cos + xx*sin^2
                        yy_rot = yy*cosVal**2 - 2*xy*sinVal*cosVal + xx*sinVal**2
                        # yz_rot becomes y_rot*z_rot = yz*cos - xz*sin
                        yz_rot = yz*cosVal - xz*sinVal
                        # zz_rot remains the same
                        zz_rot = zz
                        # Apply these changes
                        listData_rot[i][:, 0], listData_rot[i][:, 1], listData_rot[i][:, 2] = xx_rot, xy_rot, xz_rot
                        # For 6 component symmetric data
                        if nComponent == 6:
                            listData_rot[i][:, 3], listData_rot[i][:, 4] = yy_rot, yz_rot
                            listData_rot[i][:, 5] = zz_rot
                        # For 9 component symmetric data
                        elif nComponent == 9:
                            listData_rot[i][:, 3], listData_rot[i][:, 4], listData_rot[i][:, 5] = xy_rot, yy_rot, yz_rot
                            listData_rot[i][:, 6], listData_rot[i][:, 7], listData_rot[i][:, 8] = xz_rot, yz_rot, zz_rot

                # Lastly, reshape transformed data i back to old shape while replacing old values with the transformed one
                listDataRot_oldShapes[i] = listData_rot[i].reshape(np.array(listDataRot_oldShapes[i]).shape)

            return listDataRot_oldShapes

        listDataRot_oldShapes = __transform(list_data, rotateXY, dependencies)
        if len(listDataRot_oldShapes) == 1:
            listDataRot_oldShapes = listDataRot_oldShapes[0]

        return listDataRot_oldShapes

    @timer
    def readCellCenterCoordinates(self):
        ccx, ccy, ccz, cc = [], [], [], []
        for i in range(len(self.times)):
            try:
                # cc has to be in the order of x, y, z
                ccx = of.parse_internal_field(self.case_time_fullpaths[self.times[i]] + self.cc[0])
                ccy = of.parse_internal_field(self.case_time_fullpaths[self.times[i]] + self.cc[1])
                ccz = of.parse_internal_field(self.case_time_fullpaths[self.times[i]] + self.cc[2])
                cc = np.vstack((ccx, ccy, ccz)).T
                if self.save:
                    self.savePickleData(self.times[i], cc, filenames = 'CC')

                break
            except:
                pass

        if ccx is None:
            warn('\nCell centers not found! They have to be stored in at least one of {}'.format(self.times),
                 stacklevel=2)

        return ccx, ccy, ccz, cc

    @staticmethod
    @timer
    @njit(parallel=True)
    def confineFieldDomain(x, y, z, vals, bnd_x=(None, None), bnd_y=(None, None), bnd_z=(None, None), rot_z=0.):
        # assert isinstance(bnd_x, (list, tuple))
        # assert isinstance(bnd_y, (list, tuple))
        # assert isinstance(bnd_z, (list, tuple))
        # Change all None to either min or max values of the domain
        for i in prange(2):
            # For the lower bound
            if i == 0:
                bnd_x[i] = np.min(x) if bnd_x[i] is None else bnd_x[i]
                bnd_y[i] = np.min(y) if bnd_y[i] is None else bnd_y[i]
                bnd_z[i] = np.min(z) if bnd_z[i] is None else bnd_z[i]
            # For the upper bound
            else:
                bnd_x[i] = np.max(x) if bnd_x[i] is None else bnd_x[i]
                bnd_y[i] = np.max(y) if bnd_y[i] is None else bnd_y[i]
                bnd_z[i] = np.max(z) if bnd_z[i] is None else bnd_z[i]

        # for xVal, yVal, zVal, val in zip(x, y, z, vals):
        xnew, ynew, znew, valsnew = [], [], [], []
        for i in prange(len(x)):
            if x[i] >= bnd_x[0] and x[i] <= bnd_x[1] \
                and y[i] >= bnd_y[0] and y[i] <= bnd_y[1] \
                and z[i] >= bnd_z[0] and z[i] <= bnd_z[1]:
                xnew.append(x[i])
                ynew.append(y[i])
                znew.append(z[i])
                valsnew.append(vals[i])

        return xnew, ynew, znew, valsnew

    @timer
    def confineFieldDomain_Rotated(self, x, y, z, vals, box_l, box_w, box_h, box_orig=(0., 0., 0.), rot_z=0., 
                                   vals_name='data', filenames_sub='Confined', saveto_time='last'):
        print('\nConfining field domain with rotated box...')
        # Create the bounding box
        box = path.Path(((box_orig[0], box_orig[1]),
                         (box_l*np.cos(rot_z) + box_orig[0], box_l*np.sin(rot_z) + box_orig[1]),
                         (box_l*np.cos(rot_z) + box_orig[0] - box_w*np.sin(rot_z), box_l*np.sin(rot_z) + box_orig[1] + box_w*np.cos(rot_z)),
                         (box_orig[0] - box_w*np.sin(rot_z), box_orig[1] + box_w*np.cos(rot_z)),
                         (box_orig[0], box_orig[1])))
        # First confine within z range
        xnew, ynew, znew, valsnew = [], [], [], []
        cnt, milestone = 0, 25
        for i in range(len(x)):
            if z[i] < box_h + box_orig[2] and z[i] > box_orig[2]:
                xnew.append(x[i])
                ynew.append(y[i])
                znew.append(z[i])
                valsnew.append(vals[i])

        # Then confine within x and y range
        xy = (np.vstack((xnew, ynew))).T
        # Bool flags
        flags = box.contains_points(xy)
        xnew2, ynew2, znew2, valsnew2 = [], [], [], []
        for i in range(len(flags)):
            if flags[i]:
                xnew2.append(xnew[i])
                ynew2.append(ynew[i])
                znew2.append(znew[i])
                valsnew2.append(valsnew[i])

            # Gauge progress
            cnt += 1
            progress = cnt/(len(flags) + 1)*100.
            if progress >= milestone:
                print(' ' + str(milestone) + '%...', end = '')
                milestone += 25

        xnew2, ynew2, znew2, valsnew2 = np.array(xnew2), np.array(ynew2), np.array(znew2), np.array(valsnew2)
        ccnew2 = np.vstack((xnew2, ynew2, znew2)).T
        # Save ensemble field if requested
        if self.save:
            # Which time is this mean performed
            saveto_time = str(self.times[-1]) if saveto_time == 'last' else str(saveto_time)
            # Save confined cell centers
            pickle.dump(ccnew2, open(self.result_paths[saveto_time] + 'CC_' + filenames_sub + '.p', 'wb'), protocol=self.save_protocol)
            # Save confined field ensemble
            pickle.dump(valsnew2, open(self.result_paths[saveto_time] + vals_name + '_' + filenames_sub + '.p', 'wb'), protocol=self.save_protocol)
            print('\n{0} and {1} saved at {2}'.format('CC_' + filenames_sub, vals_name + '_' + filenames_sub, self.result_paths[saveto_time]))

        return xnew2, ynew2, znew2, ccnew2, valsnew2, box, flags

    @staticmethod
    @timer
    @deprecated
    def interpolateFieldData(x, y, z, vals, precisionX = 1500j, precisionY = 1500j, precisionZ = 500j, interpMethod = 'linear'):
        print('\nInterpolating field data...')
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1.000001, 0.999999)
        knownPts = np.vstack((x, y, z)).T
        # Interpolate x, y, z to designated precisions
        x3D, y3D, z3D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX,
                        y.min()*bnd[0]:y.max()*bnd[1]:precisionY,
                        z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
        # requestPts = np.vstack((x3D.ravel(), y3D.ravel(), z3D.ravel())).T
        requestPts = (x3D.ravel(), y3D.ravel(), z3D.ravel())
        dim = vals.shape
        orders, milestone, cnt = [], 2, 0
        # If vals is 1D, i.e. scalar field
        if len(dim) == 1:
            print('\nInterpolating scalar field...')
            valsND = griddata(knownPts, vals, requestPts, method = interpMethod)
        # If vals is 2D
        elif len(dim) == 2:
            # Initialize nRow x nCol x nComponent array valsND by interpolating 1st component of the values, x, or xx,
            # as others come later in the loop below
            valsND = griddata(knownPts, vals[:, 0].ravel(), requestPts, method = interpMethod)
            print('good')
            # Then go through the rest components and stack them in 4D
            for i in prange(1, vals.shape[1]):
                print(i)
                # Add i to a list in case prange is not ordered
                orders.append[i]
                # For one component
                valsND_i = griddata(knownPts, vals[:, i].ravel(), requestPts, method = interpMethod)
                # Then stack them as the 4th D
                valsND = np.dstack((valsND, valsND_i))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[1] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]
        # If vals is 3D or more
        else:
            print('good')
            # If the vals is 4D or above, reduce it to 3D
            if len(dim) > 3:
                print('better')
                vals = np.reshape(vals, (vals.shape[0], vals.shape[1], -1))
                print(vals.shape)

            # Initialize nRow x nCol x nComponent array valsND by interpolating 1st component of the values, x, or xx,
            # as others come later in the loop below
            valsND = griddata(knownPts, vals[:, :, 0].ravel(), requestPts, method = interpMethod)
            # Then go through the rest components and stack them in 4D
            for i in prange(1, vals.shape[2]):
                print(i)
                # Add i to a list in case prange is not ordered
                orders.append[i]
                # For one component
                valsND_i = griddata(knownPts, vals[:, :, i].ravel(), requestPts, method = interpMethod)
                # Then stack them as the 4th D
                valsND = np.dstack((valsND, valsND_i))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[2] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]

        # If vals was 4D, then valsND should be 5D
        if len(dim) == 4:
            valsND = np.reshape(valsND, (valsND.shape[0], valsND.shape[1], dim[2], dim[3]))

        return x3D, y3D, z3D, valsND

    @staticmethod
    @timer
    @deprecated
    def interpolateFieldData_RBF(x, y, z, vals, precisionX = 1500j, precisionY = 1500j, precisionZ = 500j,
                             function = 'linear'):
        print('\nInterpolating field data...')
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1.000001, 0.999999)
        # knownPts = np.vstack((x, y, z)).T
        # Interpolate x, y, z to designated precisions
        x3D, y3D, z3D = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precisionX,
                        y.min()*bnd[0]:y.max()*bnd[1]:precisionY,
                        z.min()*bnd[0]:z.max()*bnd[1]:precisionZ]
        # requestPts = np.vstack((x3D.ravel(), y3D.ravel(), z3D.ravel())).T
        dim = vals.shape
        orders, milestone, cnt = [], 2, 0
        if len(dim) == 1:
            rbf = Rbf(x, y, z, vals, function = function)
            valsND = rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())
        elif len(dim) == 2:
            rbf = Rbf(x, y, z, vals[:, 0], function = function)
            valsND = rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())
            for i in prange(1, dim[1]):
                print(i)
                orders.append(i)
                rbf = Rbf(x, y, z, vals[:, i], function = function)
                valsND = np.vstack((valsND, rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[2] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]
        else:
            # If the vals is 4D or above, reduce it to 3D
            if len(dim) > 3:
                vals = np.reshape(vals, (vals.shape[0], vals.shape[1], -1))

            rbf = Rbf(x, y, z, vals[:, :, 0], function = function)
            valsND = rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())
            for i in prange(1, dim[1]):
                print(i)
                orders.append(i)
                rbf = Rbf(x, y, z, vals[:, :, i], function = function)
                valsND = np.vstack((valsND, rbf(x3D.ravel(), y3D.ravel(), z3D.ravel())))
                # Gauge progress
                cnt += 1
                progress = cnt/(vals.shape[2] + 1)*100.
                if progress >= milestone:
                    print(' ' + str(milestone) + '%...', end = '')
                    milestone += 2

            print(orders)
            valsND = valsND[:, :, :, np.array(orders)]

        return x3D, y3D, z3D, valsND

    @deprecated
    # Only for uniform mesh
    def getUniformMeshInformation(self, ccx, ccy, ccz):
        from Utilities import getArrayStepping
        # Mesh size in x
        valOld = ccx[0]
        for i, val in enumerate(ccx[1:]):
            if val < valOld:
                meshSizeX = i + 1
                break

            valOld = val
            # meshSizeX = i + 2

        # Mesh size in y
        i = meshSizeX
        valOld = ccy[0]
        count = 1
        while i < ccy.size:
            if ccy[i] < valOld:
                meshSizeY = count
                break

            valOld = ccy[i]
            i += meshSizeX
            # meshSizeY = count
            count += 1

        # meshSizeY = int((np.argmax(ccy) + 1)/meshSizeX)
        # meshSizeY = np.count_nonzero(ccz == min(ccz))/meshSizeX
        meshSize = (meshSizeX, meshSizeY, int(ccz.size/(meshSizeX*meshSizeY)))

        # 1D to 3D array, access by [iX, iY, iZ]
        # List of x values: ccx3D[:, 0, 0]
        # No clue, don't ask why
        ccx3D = ccx.reshape((meshSize[2], meshSize[1], meshSize[0])).T
        ccy3D = ccy.reshape((meshSize[2], meshSize[1], meshSize[0])).T
        ccz3D = ccz.reshape((meshSize[2], meshSize[1], meshSize[0])).T

        diffMinX, diffMinY, diffMinZ = \
            getArrayStepping(ccx3D[:, 0, 0]), \
            getArrayStepping(ccy3D[0, :, 0]), \
            getArrayStepping(ccz3D[0, 0, :])
        cellSizeMin = (diffMinX, diffMinY, diffMinZ)

        print('\nMesh info retrieved')
        return meshSize, cellSizeMin, ccx3D, ccy3D, ccz3D

    @deprecated
    def convertScalarFieldToMeshGrid(self, scalarField, meshSize=None):
        scalarField = np.array(scalarField)
        if len(scalarField.shape) > 2:
            warn('\nProvided scalarField can only have two dimensions or less! Not converting to mesh grid!\n', 
                 stacklevel = 2)
            return scalarField
        
        # If meshSize is not provided, infer
        if meshSize is None:
            ccx, ccy, ccz, _ = self.readCellCenterCoordinates()
            meshSize, _, _, _, _ = self.getUniformMeshInformation(ccx, ccy, ccz)

        # scalarField3D is has the index of [x, y, z]
        scalarField3D = scalarField.reshape((meshSize[2], meshSize[1], meshSize[0])).T
        print('\nConverted scalar field to mesh grid')
        return scalarField3D


    def createTemporalMeanFields(self):
        if isinstance(self.times, (float, int)):
            warn('\nOnly one time provided, no temporal mean is calculated!\n', stacklevel = 2)
            field_data = self.readFieldData()
            return field_data

        field_data_mean = {}
        for field in self.fields:
            itime = 0
            # First mean is first time itself
            field_data_mean[field] = of.parse_internal_field(self.case_time_fullpaths[self.times[itime]] + field)
            while itime < (len(self.times) - 1):
                field_data_time = of.parse_internal_field(self.case_time_fullpaths[self.times[itime + 1]] + field)
                field_data_mean[field] += field_data_time
                itime += 1

            field_data_mean[field] /= len(self.times)

        print('\nMean field data created')
        return field_data_mean

    def decomposedFields(self, vectorfield):
        # Decompose the vector field into x, y, z, and xy components
        vectorfield = np.array(vectorfield)
        if len(vectorfield.shape) != 2:
            warn('\nThe shape of the vectorField has to be two dimensions. Do not provide mesh grid data! No '
                 'decomposition is done!\n',
                 stacklevel=2)
            return vectorfield

        if vectorfield.shape[0] == 3:
            vectorfield = vectorfield.T

        field_x, field_y, field_z, field_xy = \
            np.zeros((vectorfield.shape[0], 1)), \
            np.zeros((vectorfield.shape[0], 1)), \
            np.zeros((vectorfield.shape[0], 1)), \
            np.zeros((vectorfield.shape[0], 1))
        for i, row in enumerate(vectorfield):
            field_x[i], field_y[i], field_z[i] = row[0], row[1], row[2]
            field_xy[i] = np.sqrt(row[0]**2 + row[1]**2)

        print('\nDecomposed vector field to horizontal and vertical components')
        return field_x, field_y, field_z, field_xy
    
    def getPlanerFluctuations(self, field_xy, field_z):
        field_xy, field_z = np.array(field_xy), np.array(field_z)
        if (len(field_xy.shape) != 3) or (len(field_z.shape) != 3):
            warn('\nfield_xy and field_z have to have three indices in the order of [iX, iY, iZ]! Not creating '
                 'spatial '
                 'mean '
                 'fields!\n', stacklevel=2)
            return field_xy, field_z
        
        field_xy_mean, field_z_mean = \
            np.zeros((field_xy.shape[2], 1)), np.zeros((field_z.shape[2], 1))
        field_xy_fluc, field_z_fluc = field_xy.copy(), field_z.copy()
        for i in range(field_xy.shape[2]):
            field_xy_mean[i] = field_xy[:, :, i].mean()
            field_z_mean[i] = field_z[:, :, i].mean()
            field_xy_fluc[:, :, i] -= field_xy_mean[i]
            field_z_fluc[:, :, i] -= field_z_mean[i]

        print('\nFinished planer fluctuation fields calculation')
        return field_xy_fluc, field_z_fluc, field_xy_mean, field_z_mean
            
    @timer
    @deprecated
    def getMeanDissipationRateField(self, epsilonSGSmean, nuSGSmean, nu=1e-5, saveto_time='last'):
        # FIXME: DEPRECATED
        # According to Eq 5.64 - Eq 5.68 of Sagaut (2006), for isotropic homogeneous turbulence,
        # <epsilon> = <epsilon_resolved> + <epsilon_SGS>,
        # <epsilon_resolved>/<epsilon_SGS> = 1/(1 + (<nu_SGS>/nu)),
        # where epsilon is the total turbulence dissipation rate (m^2/s^3); and <> is statistical averaging

        epsilonMean = epsilonSGSmean/(1. - (1./(1. + nuSGSmean/nu)))
        # Avoid FPE
        epsilonMean[epsilonMean == np.inf] = 1e10
        epsilonMean[epsilonMean == -np.inf] = -1e10
        # Save to pickle if requested
        if self.save:
            # Which time is this mean performed
            saveto_time = str(self.times[-1]) if saveto_time == 'last' else str(saveto_time)
            self.savePickleData(saveto_time, epsilonMean, 'epsilonMean')
            
        return epsilonMean

    @timer
    def getStrainAndRotationRateTensorField(self, grad_u, tke=None, eps=None, cap=10.):
        """
        From Ling et al. TBNN
        :param grad_u:
        :type grad_u:
        :param tke:
        :type tke:
        :param eps:
        :type eps:
        :param cap:
        :type cap:
        :return:
        :rtype:
        """

        # Indices
        ij_uniq = self.ij_uniq
        ii6, ii9 = self.ii6, self.ii9
        ij_6to9 = self.ij_6to9
        # If either TKE or epsilon is None, no non-dimensionalization is done
        if  tke is None or eps is None:
            tke = np.ones(grad_u.shape[0])
            eps = np.ones(grad_u.shape[0])

        # Cap epsilon to 1e-10 to avoid FPE, also assuming no back-scattering
        eps[eps == 0.] = 1e-10
        # Non-dimensionalization coefficient for strain and rotation rate tensor
        tke_eps = tke/eps
        # sij is strain rate tensor, rij is rotation rate tensor
        # sij is symmetric tensor, thus 6 unique components, while rij is anti-symmetric and 9 unique components
        sij = np.empty((grad_u.shape[0], 6))
        rij = np.empty((grad_u.shape[0], 9))
        # Go through each point
        for i in prange(grad_u.shape[0]):
            grad_u_i = grad_u[i].reshape((3, 3)) if len(grad_u.shape) == 2 else grad_u[i]
            # Basically sij = 0.5TKE/epsilon*(grad_u_i + grad_u_j) that has 0 trace
            sij_i = (tke_eps[i]*0.5*(grad_u_i + grad_u_i.T)).ravel()

            # Basically rij = 0.5TKE/epsilon*(grad_u_i - grad_u_j) that has 0 in the diagonal
            rij_i = (tke_eps[i]*0.5*(grad_u_i - grad_u_i.T)).ravel()
            sij[i] = sij_i[ij_uniq]
            rij[i] = rij_i

        # Maximum and minimum
        maxsij, maxrij = np.amax(sij.ravel()), np.amax(rij.ravel())
        minsij, minrij = np.amin(sij.ravel()), np.amin(rij.ravel())
        print(' Max of sij is ' + str(maxsij) + ', and of rij is ' + str(maxrij) + ' capped to ' + str(cap))
        print(' Min of sij is ' + str(minsij) + ', and of rij is ' + str(minrij)  + ' capped to ' + str(-cap))
        sij[sij > cap], rij[rij > cap] = cap, cap
        sij[sij < -cap], rij[rij < -cap] = -cap, -cap
        # Because we enforced limits on sij, we need to re-enforce trace of 0.
        # Go through each point
        if any((maxsij > cap, minsij < cap)):
            for i in prange(grad_u.shape[0]):
                # Recall sij is symmetric and has 6 unique components
                sij[i, ii6] -=  ((1/3.*np.eye(3)*np.trace(sij[i, ij_6to9].reshape((3, 3)))).ravel()[ii9])

        return sij, rij

    @timer
    def getInvariantBasesField(self, sij, rij, quadratic_only=False, is_scale=False, zero_trace=False):
        """
        From Ling et al. TBNN
        Given sij and rij, it calculates the tensor basis
        :param sij: normalized strain rate tensor
        :param rij: normalized rotation rate tensor
        :param quadratic_only: True if only linear and quadratic terms are desired.  False if full basis is desired.
        :return: T_flat: num_points X 6 X num_tensor_basis numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3)
        >>> B[0, 1, 0] = 3.0
        >>> B[0, 0, 1] = -3.0
        >>> tb = PostProcess_FieldData.getInvariantBasisFeatures(A, B, is_scale=False)
        >>> print tb[0, :, :]
        [[  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -3.   0.   0.   0.  -3.   0.   0.   0.   6.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -6.   0.   0.   0.  -6.   0.   0.   0.  12.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -6.   0.   0.   0.  -6.   0.   0.   0.  12.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]]
        """
        def __getInvariantBasesField(sij, rij, quadratic_only, is_scale, zero_trace):
            # Indices
            ij_uniq = [0, 1, 2, 4, 5, 8]
            ij_6to9 = [0, 1, 2, 1, 3, 4, 2, 4, 5]

            # If 3D flow, then 10 tensor bases; else if 2D flow, then 4 tensor bases
            num_tensor_basis = 10 if not quadratic_only else 4
            # Tensor bases is nPoint x nBasis x 3 x 3
            # tb = np.zeros((sij.shape[0], num_tensor_basis, 3, 3))
            tb = np.empty((sij.shape[0], 6, num_tensor_basis))
            # Go through each point
            for i in prange(sij.shape[0]):
                # sij only has 6 unique components, convert it to 9 using ij_6to9
                sij_i = sij[i, ij_6to9].reshape((3, 3))
                # rij has 9 unique components already
                rij_i = rij[i].reshape((3, 3))
                # Convenient pre-computations
                sijrij = sij_i @ rij_i
                rijsij = rij_i @ sij_i
                sijsij = sij_i @ sij_i
                rijrij = rij_i @ rij_i
                # 10 tensor bases for each point and each (unique) bij component
                # 1: sij
                tb[i, :, 0] = sij_i.ravel()[ij_uniq]
                # 2: SijRij - RijSij
                tb[i, :, 1] = (sijrij - rijsij).ravel()[ij_uniq]
                # 3: sij^2 - 1/3I*tr(sij^2)
                tb[i, :, 2] = (sijsij - 1./3.*np.eye(3)*np.trace(sijsij)).ravel()[ij_uniq]
                # 4: rij^2 - 1/3I*tr(rij^2)
                tb[i, :, 3] = (rijrij - 1./3.*np.eye(3)*np.trace(rijrij)).ravel()[ij_uniq]
                if not quadratic_only:
                    # 5: RijSij^2 - sij^2Rij
                    tb[i, :, 4] = (rij_i @ sijsij - sij_i @ sijrij).ravel()[ij_uniq]
                    # 6: rij^2Sij + SijRij^2 - 2/3I*tr(SijRij^2)
                    tb[i, :, 5] = (rij_i @ rijsij
                                    + sij_i @ rijrij
                                    - 2./3.*np.eye(3)*np.trace(sij_i @ rijrij)).ravel()[ij_uniq]
                    # 7: RijSijRij^2 - rij^2SijRij
                    tb[i, :, 6] = (rijsij @ rijrij - rijrij @ sijrij).ravel()[ij_uniq]
                    # 8: SijRijSij^2 - sij^2RijSij
                    tb[i, :, 7] = (sijrij @ sijsij - sijsij @ rijsij).ravel()[ij_uniq]
                    # 9: rij^2Sij^2 + sij^2Rij^2 - 2/3I*tr(sij^2Rij^2)
                    tb[i, :, 8] = (rijrij @ sijsij
                                    + sijsij @ rijrij
                                    - 2./3.*np.eye(3)*np.trace(sijsij @ rijrij)).ravel()[ij_uniq]
                    # 10: RijSij^2Rij^2 - rij^2Sij^2Rij
                    tb[i, :, 9] = ((rij_i @ sijsij) @ rijrij
                                    - (rij_i @ rijsij) @ sijrij).ravel()[ij_uniq]

                # Enforce zero trace for anisotropy for each basis
                if zero_trace:
                    for j in range(num_tensor_basis):
                        # Recall tb is shape (n_samples, 6, n_bases)
                        tb[i, :, j] -= (1./3.*np.eye(3)*np.trace(tb[i, ij_6to9, j].reshape((3, 3)))).ravel()[ij_uniq]

            # Scale down to promote convergence
            if is_scale:
                # Using tuple gives Numba error
                scale_factor = [1, 10, 10, 10, 100, 100, 1000, 1000, 1000, 1000]
                # Go through each basis
                for j in range(1, num_tensor_basis):
                    tb[:, :, j] /= scale_factor[j]

            return tb

        tb = __getInvariantBasesField(sij, rij, quadratic_only, is_scale, zero_trace)

        return tb

    @timer
    def calcScalarBasis(self, sij, rij, is_train=False, cap=100.0, is_scale=True):
        """
        Given the non-dimensionalized mean strain rate and mean rotation rate tensors sij and rij,
        this returns a set of normalized scalar invariants
        :param sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
        :param rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3) * 2.0
        >>> B[0, 1, 0] = 1.0
        >>> B[0, 0, 1] = -1.0
        >>> tdp = TurbulenceKEpsDataProcessor()
        >>> tdp.mu = 0
        >>> tdp.std = 0
        >>> scalar_basis = tdp.calc_scalar_basis(A, B, is_scale=False)
        >>> print scalar_basis
        [[ 12.  -2.  24.  -4.  -8.]]
        """
        if is_train:
            self.mu = None
            self.std = None
            
        num_points = sij.shape[0]
        num_invariants = 5
        invariants = np.zeros((num_points, num_invariants))
        for i in prange(num_points):
            invariants[i, 0] = np.trace(np.dot(sij[i, :, :], sij[i, :, :]))
            invariants[i, 1] = np.trace(np.dot(rij[i, :, :], rij[i, :, :]))
            invariants[i, 2] = np.trace(np.dot(sij[i, :, :], np.dot(sij[i, :, :], sij[i, :, :])))
            invariants[i, 3] = np.trace(np.dot(rij[i, :, :], np.dot(rij[i, :, :], sij[i, :, :])))
            invariants[i, 4] = np.trace(np.dot(np.dot(rij[i, :, :], rij[i, :, :]), np.dot(sij[i, :, :], sij[i, :, :])))

        # Renormalize invariants using mean and standard deviation:
        if is_scale:
            if self.mu is None or self.std is None:
                is_train = True

            if is_train:
                self.mu = np.zeros((num_invariants, 2))
                self.std = np.zeros((num_invariants, 2))
                self.mu[:, 0] = np.mean(invariants, axis=0)
                self.std[:, 0] = np.std(invariants, axis=0)

            invariants = (invariants - self.mu[:, 0])/self.std[:, 0]
            maxInvariants, minInvariants = np.amax(invariants), np.amin(invariants)
            print(' Max of scaled scalar basis is {}'.format(maxInvariants))
            print(' Max of scaled scalar basis is {}'.format(minInvariants))
            # Why cap?????
            invariants[invariants > cap] = cap  # Cap max magnitude
            invariants[invariants < -cap] = -cap
            invariants = invariants*self.std[:, 0] + self.mu[:, 0]
            if is_train:
                self.mu[:, 1] = np.mean(invariants, axis=0)
                self.std[:, 1] = np.std(invariants, axis=0)

            invariants = (invariants - self.mu[:, 1])/self.std[:, 1]  # Renormalize a second time after capping
        return invariants, self.mu, self.std

    @timer
    def getAnisotropyTensorField(self, stress, use_oldshape=True):
        # Reshape u'u' to 2D, with nPoint x 6/9
        shape_old = stress.shape
        # If u'u' is 4D, then assume first 2D are mesh grid and last 2D are 3 x 3 and reshape to nPoint x 9
        if len(stress.shape) == 4:
            stress = stress.reshape((stress.shape[0]*stress.shape[1], 9))
        # Else if u'u' is 3D
        elif len(stress.shape) == 3:
            # If 3rd D has 3, then assume nPoint x 3 x 3 and reshape to nPoint x 9
            if stress.shape[2] == 3:
                stress = stress.reshape((stress.shape[0], 9))
            # Else if 3rd D has 6, then assume nX x nY x 6 and reshape to nPoint x 9
            elif stress.shape[2] == 6:
                stress = stress.reshape((stress.shape[0]*stress.shape[1], 6))
            # Else if 3rd D has 9, then assume nX x nY x 9 and reshape to nPoint x 9
            elif stress.shape[2] == 9:
                stress = stress.reshape((stress.shape[0]*stress.shape[1], 9))

        # If u'u' is provided as a symmetric tensor
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        # Otherwise, xx is '0', yy is '4', zz is '8'
        xx, yy, zz = (0, 3, 5) if stress.shape[1] == 6 else (0, 4, 8)
        # TKE
        k = 0.5*(stress[:, xx] + stress[:, yy] + stress[:, zz])
        # Avoid FPE
        k[k < 1e-10] = 1e-10
        # Convert u'u' to bij
        bij = np.empty_like(stress)
        for i in prange(stress.shape[1]):
            bij[:, i] = stress[:, i]/(2.*k) - 1/3. if i in (xx, yy, zz) else stress[:, i]/(2.*k)

        # Reshape bij back to initial provide shape
        if bij.shape != shape_old and use_oldshape: bij = bij.reshape(shape_old)

        return bij

    @timer
    @deprecated
    def evaluateInvariantBasisCoefficients(self, tb, bij, cap=100., onegToRuleThemAll=False, saveto_time='last'):
        # def __getInvariantBasisCoefficientsField(tb, bij, onegToRuleThemAll):
        # If tensor bases is 4D, i.e. nPoint x nBasis x 3 x 3,
        # then reshape it to 3D, i.e. nPoint x nBasis x 9
        tb  = tb.reshape((tb.shape[0], tb.shape[1], 9)) if len(tb.shape) == 4 else tb
        # If bij is 3D, i.e. nPoint x 3 x 3 / nPoint x 1 x 3 x 3,
        # do the same thing so bij is nPoint x 9
        bij = bij.reshape((bij.shape[0], 9)) if len(bij.shape) in (3, 4) else bij
        # If one set of 10 g to describe all points
        if onegToRuleThemAll:
            # Sum of all TB per point, so 0 point x nBasis x 9 i.e. nBasis x 9 (Numpy sum does dimension reduction)
            tb_sum = np.sum(tb, axis = 0)
            # Do the same for bij, so shape (9,)
            bij_sum = np.sum(bij, axis = 0)
            # Since, for each ij component, solving TB_ij^k*g^k = bij for g can result in a different g,
            # use least squares to solve the linear system of TB_ij^k*g^k = bij,
            # with row being 9 components and column being nBasis
            # Therefore, transpose TB first so TB.T is 9 component x nBasis, bij shape is (9,)
            tb_sum_T = tb_sum.T
            g = np.linalg.lstsq(tb_sum_T, bij_sum, rcond = None)[0]
            rmse = np.sqrt(np.mean(np.square(bij_sum - np.dot(tb_sum_T, g))))
        # Else if different 10 g for every point
        else:
            # Initialize tensor basis coefficients, nPoint x nBasis
            g, rmse = np.empty((tb.shape[0], tb.shape[1])), np.empty(tb.shape[0])
            # Go through each point
            for p in prange(tb.shape[0]):
                # Do the same as above, just for each point
                # Row being 9 components and column being nBasis
                # For each point p, tb[p].T is 9 component x nBasis, bij[p] shape is (9,)
                tb_p = tb[p].T
                g[p] = np.linalg.lstsq(tb_p, bij[p], rcond = None)[0]
                # TODO: couldn't get RMSE driectly from linalg.lstsq cuz my rank of tb[p] is 5 < 9?
                rmse[p] = np.sqrt(np.mean(np.square(bij[p] - np.dot(tb_p, g[p]))))

            # return g, rmse

        # g, rmse = __getInvariantBasisCoefficientsField(tb, bij, onegToRuleThemAll)
        # Advanced slicing is not support by nijt
        # Cap extreme values
        g[g > cap] = cap
        g[g < -cap] = cap

        # # TODO: save gives error to Numba
        # # Save data if requested
        # if self.save:
        #     saveto_time = self.times[len(self.times)] if saveto_time == 'last' else saveto_time
        #     self.savePickleData(saveto_time, g, 'g')

        return g, rmse

    def savePickleData(self, time, list_data, filenames=('data',)):
        if isinstance(list_data, np.ndarray):
            list_data = (list_data,)

        if isinstance(filenames, str):
            filenames = (filenames,)

        if len(filenames) != len(list_data):
            filenames = ['data' + str(i) for i in range(len(list_data))]
            warn('\nInsufficient filenames provided! Using default filenames...', stacklevel=2)

        for i in range(len(list_data)):
            pickle.dump(list_data[i], open(self.result_paths[str(time)] + '/' + filenames[i] + '.p', 'wb'),
                        protocol=self.save_protocol)

        print('\n{0} saved at {1}'.format(filenames, self.result_paths[str(time)]))

    def readPickleData(self, time, filenames):
        """
        Read pickle data of one or more fields at one time and return them in a dictionary
        :param time:
        :type time:
        :param filenames:
        :type filenames:
        :return:
        :rtype:
        """
        # Ensure loop-able
        if isinstance(filenames, str):
            filenames = (filenames,)

        # Encoding = 'latin1' is required for unpickling Numpy arrays pickled by Python 2
        encode = 'ASCII' if self.save_protocol >= 3 else 'latin1'
        datadict = {}
        # Go through each file
        for i in range(len(filenames)):
            datadict[filenames[i]] = pickle.load(open(self.result_paths[str(time)] + filenames[i] + '.p', 'rb'),
                                                 encoding=encode)

        # If just one file read, then no need for a dictionary
        if len(filenames) == 1:
            datadict = datadict[filenames[0]]

        print('\n{} read. If multiple files read, data is stored in dictionary'.format(filenames))
        return datadict

    @deprecated
    def createSliceData(self, field_data, baseCoordinate = (0, 0, 90), normalVector = (0, 0, 1)):
        from Utilities import takeClosest
        # Only support horizontal or vertical slices
        # meshSize has to be in the order of x, y, z
        # A plane refers to a horizontal plane
        # cellsPerPlane = self.meshSize[0]*self.meshSize[1]
        # nPlane = self.meshSize[2]

        # Vertical slice, ignore z of baseCoordinate and normalVector
        # x has to be non-negative
        if normalVector[2] == 0:
            # If x = 0, then slice angle is 0 deg and is in xz plane
            if normalVector[0] == 0:
                angleXY = 0.
            else:
                angleXY = 0.5*np.pi + np.arctan(normalVector[1]/normalVector[0])

            # Draw a line across xy-plane that passes baseCoordinate[:2]
            # y = kx + b
            k = np.tan(angleXY)
            b = baseCoordinate[1] - k*baseCoordinate[0]

            ccx, ccy, ccz, cc = self.readCellCenterCoordinates()
            # # Find out cellSize, 0.99 of real cell size to avoid machine error
            # # If smaller than 1, then set it to 1
            # cellSizeEff = max(min(0.99*(ccx[1] - ccx[0]), cellSize), 1)
            meshSize, cellSizeMin, _, ccy3D, _ = self.getMeshInfo(ccx, ccy, ccz)
            lineXs = np.arange(0, max(ccx), cellSizeMin[0])
            lineYs = k*lineXs + b

            # Find y limits
            yLims = (k*ccx[0] + b, k*ccx[meshSize[0] - 1] + b)
            ccyLow, ccyLowIdx, _ = takeClosest(ccy3D[0, :, 0], yLims[0])
            ccyHi, ccyHiIdx, _ = takeClosest(ccy3D[0, :, 0], yLims[1])

            # Find target x for each y coordinate of the cells in a horizontal plane
            xTargets = []
            i = ccyLowIdx
            while i <= ccyHiIdx:
                xTargets.append((ccy3D[0, i, 0] - b)/k)
                i += 1

            # For the lowest horizontal plane, find all x coordinates and cell indices of this vertical slice
            # cellCoorXs = []
            iCells = []
            # The start row of x, i.e. starting y index is ccyLowIdx
            rowX = ccyLowIdx
            for xTarget in xTargets:
                val, iCellX, diff = takeClosest(ccx[:meshSize[0]], xTarget)
                # cellCoorXs.append(cellCoorX)
                # Every next find is on the next row of x, i.e. next y
                # Position in a row, iCellX, + row shift, rowX*meshSize[0]
                iCells.append(iCellX + rowX*meshSize[0])
                rowX += 1

            iCells = np.array(iCells)
            # Repeat for all horizontal planes
            iPlane = 1
            # Lowest plane of vertical slice cell indices
            iCellsBase = iCells
            while iPlane < meshSize[2]:
                # Next plane of vertical slice cell indices are current plane's + a whole plane
                iCellsNext = iCellsBase + iPlane*meshSize[0]*meshSize[1]
                iCells = np.hstack((iCells, iCellsNext))
                iPlane += 1

        fieldDataSlice = np.take(field_data, indices = iCells, axis = 0)
        ccSlice = np.take(cc, indices = iCells, axis = 0)

        # Dimension of the slice, (size in diagonal, size in vertical)
        sliceDim = (len(iCellsBase), 1, iPlane)

        print('\nField data slice created')
        return fieldDataSlice, ccSlice, sliceDim




if __name__ == '__main__':
    casename = 'ALM_N_H_OneTurb'
    casedir = '/media/yluan'
    fields = 'grad_UAvg'
    time = '24995.0788025'

    case = FieldData(fields = fields, times = time, casename = casename, casedir = casedir)
    data = case.readFieldData()




            
        
