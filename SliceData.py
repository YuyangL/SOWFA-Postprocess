import numpy as np
from Utility import timer, deprecated
from numba import njit
from scipy.interpolate import griddata
import os
import PostProcess_AnisotropyTensor

class SliceProperties:
    def __init__(self, time='latestTime', casedir='/media/yluan/Toshiba External Drive', casename='ALM_N_H_ParTurb', slice_folder='Slices', result_folder='Result', rot_z=0.):
        self.case_fullpath = casedir + '/' + casename + '/' + slice_folder + '/'
        self.result_path = self.case_fullpath + result_folder + '/'
        os.makedirs(self.result_path, exist_ok=True)
        # If time is 'latestTime', find it automatically
        if 'latest' in time:
            # Excl. "Result" folder string
            times = [float(time) if '.' in time else int(time) for time in os.listdir(self.case_fullpath)[:-1]]
            self.time = str(max(times))
        else:
            self.time = str(time)
            
        # Update case full path to include time folder as well
        self.case_fullpath += self.time + '/'
        # Add the selected time folder in result path if not existent already
        self.result_path += self.time + '/'
        os.makedirs(self.result_path, exist_ok=True)
        # Orientate around z axis in case of angled flow direction
        # Angle in rad and counter-clockwise, automatic unit detection
        self.rot_z = rot_z if rot_z > 0.5*np.pi else rot_z/180.*np.pi

    @timer
    def readSlices(self, properties=('U',), slicenames=('alongWind',), slicenames_sub='Slice', skipcol=3, skiprow=0, file_ext='raw'):
        # First 3 columns are x, y, z, thus skipcol = 3
        # skiprow unnecessary since np.genfromtxt trim any header with # at front
        slicenames = (slicenames,) if isinstance(slicenames, str) else slicenames
        self.properties = (properties,) if isinstance(properties, str) else properties
        # Combine property with slicenames and subscript to form the full file names
        self.slicenames = []
        for property in self.properties:
            for slicename in slicenames:
                self.slicenames.append(property + '_' + slicename)

        self.slices_val, self.slices_orient, self.slices_coor = {}, {}, {}
        # Go through all specified slices
        # and append coordinates,, slice type (vertical or horizontal), and slice values each to dictionaries
        # Keys are slice names
        for slicename in self.slicenames:
            vals = np.genfromtxt(self.case_fullpath + slicename + '_' + slicenames_sub + '.' + file_ext)
            # If max(z) - min(z) < 1 then it's assumed horizontal
            # partition('.') removes anything after '.'
            # fileName.partition('.')[0]
            self.slices_orient[slicename] = 'vertical' if (vals[skiprow:, 2].max()
            - vals[skiprow:, 2].min() > 1.) else 'horizontal'
            # # If interpolation enabled
            # if interp_method not in ('none', 'None'):
            #     print('\nInterpolation enabled')
            #     # X2D, Y2D, Z2D, vals2D dictionary after interpolation
            #     slicesX[fileName.partition('.')[0]], slicesY[fileName.partition('.')[0]], slicesZ[fileName.partition('.')[0]], slices_val[fileName.partition('.')[0]] = \
            #         self.interpolateSliceData(vals[skiprow:, 0], vals[skiprow:, 1], vals[skiprow:, 2], vals[skiprow:, skipcol:], slice_orient = slices_orient[fileName.partition('.')[0]], precision_x = precision[0], precision_y = precision[1], precision_z = precision[2], interp_method = interp_method)
            #
            # else:
            # X, Y, Z coordinate dictionary without interpolation
            self.slices_coor[slicename] = vals[skiprow:, :skipcol]
            # Vals dictionary without interpolation
            self.slices_val[slicename] = vals[skiprow:, skipcol:]

        print('\n' + str(self.slicenames) + ' read')
        # return slices_coor, slices_orient, slices_val

    @staticmethod
    @timer
    def interpolateDecomposedSliceData_Fast(x, y, z, vals, slice_orient='vertical', target_meshsize=1e4,
                                            rot_z=0, interp_method='nearest', confinebox=None):
        # confinebox[4:6] will be ignored if slice is horizontal
        # Automatically determine best x, y, z precision
        # If no confine region specified
        if confinebox is None:
            lx = np.max(x) - np.min(x)
            # If vertical slice, ly doesn't contribute to slice interpolation, thus 0
            ly = np.max(y) - np.min(y) if slice_orient == 'horizontal' else 0
            # If horizontal slice, z length should be about 0
            lz = np.max(z) - np.min(z) if slice_orient == 'vertical' else 0
        else:
            lx = confinebox[1] - confinebox[0]
            ly = confinebox[3] - confinebox[2] if slice_orient == 'horizontal' else 0
            lz = confinebox[5] - confinebox[4] if slice_orient == 'vertical' else 0

        # Sum of all "contributing" lengths, as scaler
        lxyz = lx + ly + lz
        # Weight of each precision, minimum 0.001
        xratio = np.max((lx/lxyz, 0.001))
        # For vertical slices, the resolution for x is the same as y, i.e. target_meshsize is shared between x and z only
        yratio = np.max((ly/lxyz, 0.001)) if slice_orient == 'horizontal' else 1.
        # If horizontal slice, z cell is aimed at 1
        # zratio = 1 effectively removes the impact of z on horizontal slice resolutions
        zratio = np.max((lz/lxyz, 0.001)) if slice_orient == 'vertical' else 1.
        # Base precision
        # Using xratio*base*(yratio*base or zratio*base) = target_meshsize
        precision_base = target_meshsize/(xratio*yratio*zratio)
        # If horizontal slice, target_meshsize is shared between x and y only
        precision_base = precision_base**(1/2.)
        # Derived precision, take ceiling
        precision_x = int(np.ceil(xratio*precision_base))
        # If vertical slice, precision_y is the same as precision_x
        precision_y = int(np.ceil(yratio*precision_base)) if slice_orient == 'horizontal' else precision_x
        precision_z = int(np.ceil(zratio*precision_base)) if slice_orient == 'vertical' else 1
        print('\nInterpolating slice with precision ({0}, {1}, {2})...'.format(precision_x, precision_y, precision_z))
        precision_x *= 1j
        precision_y *= 1j
        precision_z *= 1j
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1, 1)
        # print("xmin = {}, xmax = {}".format(x.min(), x.max()))
        # print("confinebox[0] = {}, confinebox[1] = {}".format(confinebox[0], confinebox[1]))
        # print("ymin = {}, ymax = {}".format(y.min(), y.max()))
        # print("confinebox[2] = {}, confinebox[3] = {}".format(confinebox[2], confinebox[3]))
        # print("zmin = {}, zmax = {}".format(z.min(), z.max()))
        # print("confinebox[4] = {}, confinebox[5] = {}".format(confinebox[4], confinebox[5]))
        bnd_xtarget = (x.min()*bnd[0], x.max()*bnd[1]) if confinebox is None else (confinebox[0], confinebox[1])
        bnd_ytarget = (y.min()*bnd[0], y.max()*bnd[1]) if confinebox is None else (confinebox[2], confinebox[3])
        bnd_ztarget = (z.min()*bnd[0], z.max()*bnd[1]) if confinebox is None else (confinebox[4], confinebox[5])
        # If vertical slice
        if slice_orient == 'vertical':
            # Known x and z coordinates, to be interpolated later
            known_pts = np.vstack((x, z)).T
            # Interpolate x and z according to precisions
            x2d, z2d = np.mgrid[bnd_xtarget[0]:bnd_xtarget[1]:precision_x, bnd_ztarget[0]:bnd_ztarget[1]:precision_z]
            # Then interpolate y in the same fashion of x
            # Thus the same precision as x
            y2d, _ = np.mgrid[bnd_ytarget[0]:bnd_ytarget[1]:precision_x, bnd_ztarget[0]:bnd_ztarget[1]:precision_z]
            # In case the vertical slice is at a negative angle,
            # i.e. when x goes from low to high, y goes from high to low,
            # flip y2d from low to high to high to low
            y2d = np.flipud(y2d) if ((x[0] - x[1])*(y[0] - y[1])) < 0. else y2d
        # Else if slice is horizontal
        else:
            known_pts = np.vstack((x, y)).T
            x2d, y2d = np.mgrid[bnd_xtarget[0]:bnd_xtarget[1]:precision_x, bnd_ytarget[0]:bnd_ytarget[1]:precision_y]
            # For horizontal slice, z is constant, thus not affected by confinebox
            # Also z2d is like y2d, thus same precision as y
            _, z2d = np.mgrid[bnd_xtarget[0]:bnd_xtarget[1]:precision_x, z.min()*bnd[0]:z.max()*bnd[1]:precision_y]

        # Decompose the vector/tensor of slice values
        # If vector, order is x, y, z
        # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
        # Second axis to interpolate is z if vertical slice otherwise y fo horizontal slices
        grid_coor2 = z2d if slice_orient == 'vertical' else y2d
        # For gauging progress
        milestone = 33
        # If vals is 2D
        if len(vals.shape) == 2:
            # print("Min x = {}, max = {}".format(min(x2d.ravel()), max(x2d.ravel())))
            # print("Min coor2 = {}, max = {}".format(min(grid_coor2.ravel()), max(grid_coor2.ravel())))
            # Initialize nRow x nCol x nComponent array vals3d by interpolating first component of the values, x, or xx
            vals3d = np.empty((x2d.shape[0], x2d.shape[1], vals.shape[1]))
            # Then go through the rest components and stack them in 3D
            for i in range(vals.shape[1]):
                # Each component is interpolated from the known locations pointsXZ to refined fields (x2d, z2d)
                vals3d[:, :, i] = griddata(known_pts, vals[:, i], (x2d, grid_coor2), method=interp_method)
                # vals3d = np.dstack((vals3d, vals3d_i))
                # Gauge progress
                progress = (i + 1)/vals.shape[1]*100.
                if progress >= milestone:
                    print(' {0}%... '.format(milestone))
                    milestone += 33

        # Else if vals is 3D
        else:
            vals3d = np.empty((x2d.shape[0], x2d.shape[1], vals.shape[2]))
            for i in range(vals.shape[2]):
                vals3d_i = griddata(known_pts, vals[:, :, i].ravel(), (x2d, grid_coor2), method=interp_method)
                vals3d[:, :, i] = vals3d_i
                progress = (i + 1)/vals.shape[2]*100.
                if progress >= milestone:
                    print(' {0}%... '.format(milestone))
                    milestone += 33

        # if rot_z != 0:
        #     # If vector, x, y, z
        #     if vals.shape[1] == 3:
        #         vals3d['0'] = vals3d['0']*np.cos(rot_z) + vals3d['1']*np.sin(rot_z)
        #         vals3d['1'] = -vals3d['0']*np.sin(rot_z) + vals3d['1']*np.cos(rot_z)
        #     else:
        #         vals3d['0'] =

        # x2d, y2d, z2d = np.nan_to_num(x2d), np.nan_to_num(y2d), np.nan_to_num(z2d)
        # vals3d = np.nan_to_num(vals3d)

        return x2d, y2d, z2d, vals3d


    # [DEPRECATED] Refer to interpolateDecomposedSliceData_Fast()
    @staticmethod
    @timer
    @deprecated
    def interpolateDecomposedSliceData(x, y, z, vals, slice_orient = 'vertical', rot_z = 0, precision_x = 1500j, precision_y = 1500j,
                          precision_z = 500j, interp_method = 'cubic'):
        # Bound the coordinates to be interpolated in case data wasn't available in those borders
        bnd = (1.00001, 0.99999)
        if slice_orient is 'vertical':
            # Known x and z coordinates, to be interpolated later
            known_pts = np.vstack((x, z)).T
            # Interpolate x and z according to precisions
            x2d, z2d = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precision_x, z.min()*bnd[0]:z.max()*bnd[1]:precision_z]
            # Then interpolate y in the same fashion of x
            y2d, _ = np.mgrid[y.min()*bnd[0]:y.max()*bnd[1]:precision_y, z.min()*bnd[0]:z.max()*bnd[1]:precision_z]
            # In case the vertical slice is at a negative angle,
            # i.e. when x goes from low to high, y goes from high to low,
            # flip y2d from low to high to high to low
            y2d = np.flipud(y2d) if x[0] > x[1] else y2d
        else:
            known_pts = np.vstack((x, y)).T
            x2d, y2d = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precision_x, y.min()*bnd[0]:y.max()*bnd[1]:precision_y]
            _, z2d = np.mgrid[x.min()*bnd[0]:x.max()*bnd[1]:precision_x, z.min()*bnd[0]:z.max()*bnd[1]:precision_z]

        # Decompose the vector/tensor of slice values
        # If vector, order is x, y, z
        # If symmetric tensor, order is xx, xy, xz, yy, yz, zz
        valsDecomp = {}
        for i in range(vals.shape[1]):
            if slice_orient is 'vertical':
                # Each component is interpolated from the known locations pointsXZ to refined fields (x2d, z2d)
                valsDecomp[str(i)] = griddata(known_pts, vals[:, i].ravel(), (x2d, z2d), method = interp_method)
            else:
                valsDecomp[str(i)] = griddata(known_pts, vals[:, i].ravel(), (x2d, y2d), method = interp_method)

        # if rot_z != 0:
        #     # If vector, x, y, z
        #     if vals.shape[1] == 3:
        #         valsDecomp['0'] = valsDecomp['0']*np.cos(rot_z) + valsDecomp['1']*np.sin(rot_z)
        #         valsDecomp['1'] = -valsDecomp['0']*np.sin(rot_z) + valsDecomp['1']*np.cos(rot_z)
        #     else:
        #         valsDecomp['0'] =

        return x2d, y2d, z2d, valsDecomp


    @staticmethod
    @timer
    @njit(parallel=True, fastmath=True)
    def processAnisotropyTensor_Fast(vals3d):
        # TKE in the interpolated mesh
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        k = 0.5*(vals3d[:, :, 0] + vals3d[:, :, 3] + vals3d[:, :, 5])
        # Convert Rij to bij
        for i in range(6):
            vals3d[:, :, i] = vals3d[:, :, i]/(2.*k) - 1/3. if i in (0, 3, 5) else vals3d[:, :, i]/(2.*k)

        # Add each anisotropy tensor to each mesh grid location, in depth
        # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
        tensors = np.dstack((vals3d[:, :, 0], vals3d[:, :, 1], vals3d[:, :, 2],
                             vals3d[:, :, 1], vals3d[:, :, 3], vals3d[:, :, 4],
                             vals3d[:, :, 2], vals3d[:, :, 4], vals3d[:, :, 5]))
        # Reshape the z dir to 3x3 instead of 9x1
        # Now tensors is 4D, with x, y being mesh grid, z1, z2 being the 3x3 tensor at (x, y)
        tensors = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))

        # Evaluate eigenvalues of symmetric tensor
        eigValsGrid = np.zeros((1,3))
        for i in range(tensors.shape[0]):
            for j in range(tensors.shape[1]):
                # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
                eigVals = np.linalg.eigvalsh(tensors[i, j, :, :])
                # Numba njit doesn't support np.flipud, thus manually reverse eigVals to high to low
                tmpVal = eigVals[0]
                eigVals[0] = eigVals[2]
                eigVals[2] = tmpVal
                eigVals2D = eigVals.reshape((1, 3))
                # if i == 0 and j == 0:
                #     eigValsGrid0 = np.zeros_like(eigVals2D)
                #     eigValsGrid = np.vstack((eigValsGrid0, eigVals2D))
                #     print(eigValsGrid.shape)
                # else:
                # Each eigVals is a row, stack them vertically
                eigValsGrid = np.vstack((eigValsGrid, eigVals2D))

        # Reshape eigVals to nRow x nCol x 3
        # so that each mesh grid location has 3 eigenvalues
        # Remove the first row since it was dummy
        eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))

        return vals3d, tensors, eigValsGrid


    # [DEPRECATED] Refer to PostProcess_AnisotropyTensor
    @staticmethod
    @timer
    @njit(parallel=True, fastmath=True)
    def processAnisotropyTensor_Uninterpolated(vals2D):
        # TKE
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        k = 0.5*(vals2D[:, 0] + vals2D[:, 3] + vals2D[:, 5])
        # Convert Rij to bij
        for i in range(6):
            vals2D[:, i] = vals2D[:, i]/(2.*k) - 1/3. if i in (0, 3, 5) else vals2D[:, i]/(2.*k)

        # Add each anisotropy tensor to each mesh grid location, in depth
        # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
        tensors = np.dstack((vals2D[:, 0], vals2D[:, 1], vals2D[:, 2],
                             vals2D[:, 1], vals2D[:, 3], vals2D[:, 4],
                             vals2D[:, 2], vals2D[:, 4], vals2D[:, 5]))
        # Reshape the z dir to 3x3 instead of 9x1
        # Now tensors is 4D, with x, y being nRow, 1, z1, z2 being the 3x3 tensor at (x, y)
        # Don't know why x, y shape has to be flipped?
        # tensors1 = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))
        tensors = tensors.reshape((tensors.shape[1], tensors.shape[0], 3, 3))
        # print(tensors1[0, 100, :])
        # print(tensors[100, 0, :])

        # Evaluate eigenvalues of symmetric tensor
        eigValsGrid = np.zeros((1, 3))
        for i in range(tensors.shape[0]):
            for j in range(tensors.shape[1]):
                # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
                eigVals = np.linalg.eigvalsh(tensors[i, j, :, :])
                # Numba njit doesn't support np.flipud, thus manually reverse eigVals to high to low
                tmpVal = eigVals[0]
                eigVals[0] = eigVals[2]
                eigVals[2] = tmpVal
                eigVals2D = eigVals.reshape((1, 3))
                # if i == 0 and j == 0:
                #     eigValsGrid0 = np.zeros_like(eigVals2D)
                #     eigValsGrid = np.vstack((eigValsGrid0, eigVals2D))
                #     print(eigValsGrid.shape)
                # else:
                # Each eigVals is a row, stack them vertically
                eigValsGrid = np.vstack((eigValsGrid, eigVals2D))

        # Reshape eigVals to nRow x nCol x 3
        # so that each mesh grid location has 3 eigenvalues
        # Remove the first row since it was dummy
        eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))

        return vals2D, tensors, eigValsGrid


    @staticmethod
    @timer
    def processAnisotropyTensor(valsDecomp):
        # TKE in the interpolated mesh
        # xx is '0', xy is '1', xz is '2', yy is '3', yz is '4', zz is '5'
        k = 0.5*(valsDecomp['0'] + valsDecomp['3'] + valsDecomp['5'])
        # Convert Rij to bij
        for key, val in valsDecomp.items():
            valsDecomp[key] = val/(2.*k) - 1/3. if key in ('0', '3', '5') else val/(2.*k)

        # Add each anisotropy tensor to each mesh grid location, in depth
        # tensors is 3D with z being b11, b12, b13, b21, b22, b23...
        tensors = np.dstack((valsDecomp['0'], valsDecomp['1'], valsDecomp['2'],
                          valsDecomp['1'], valsDecomp['3'], valsDecomp['4'],
                          valsDecomp['2'], valsDecomp['4'], valsDecomp['5']))
        # Reshape the z dir to 3x3 instead of 9x1
        # Now tensors is 4D, with x, y being mesh grid, z1, z2 being the 3x3 tensor at (x, y)
        tensors = tensors.reshape((tensors.shape[0], tensors.shape[1], 3, 3))

        # Evaluate eigenvalues of symmetric tensor
        eigValsGrid = [0, 0, 0]
        for i in range(tensors.shape[0]):
            for j in range(tensors.shape[1]):
                # eigVals is in ascending order, reverse it so that lambda1 >= lambda2 >= lambda3
                eigVals = np.flipud(np.linalg.eigvalsh(tensors[i, j, :, :]))
                # Each eigVals is a row, stack them vertically
                eigValsGrid = np.vstack((eigValsGrid, eigVals))

        # Reshape eigVals to nRow x nCol x 3
        # so that each mesh grid location has 3 eigenvalues
        # Remove the first row since it was dummy
        eigValsGrid = np.reshape(eigValsGrid[1:], (tensors.shape[0], tensors.shape[1], 3))

        return valsDecomp, tensors, eigValsGrid


    @staticmethod
    @timer
    def getBarycentricMapCoordinates(eigValsGrid, c_offset = 0.65, c_exp = 5.):
        # Coordinates of the anisotropy tensor in the tensor basis {a1c, a2c, a3c}. From Banerjee (2007),
        # C1c = lambda1 - lambda2,
        # C2c = 2(lambda2 - lambda3),
        # C3c = 3lambda3 + 1
        c1 = eigValsGrid[:, :, 0] - eigValsGrid[:, :, 1]
        # Not used for coordinates, only for color maps
        c2 = 2*(eigValsGrid[:, :, 1] - eigValsGrid[:, :, 2])
        c3 = 3*eigValsGrid[:, :, 2] + 1
        # Corners of the barycentric triangle
        # Can be random coordinates?
        x1c, x2c, x3c = 1., 0., 1/2.
        y1c, y2c, y3c = 0, 0, np.sqrt(3)/2
        # xBary, yBary = c1*x1c + c2*x2c + c3*x3c, c1*y1c + c2*y2c + c3*y3c
        xBary, yBary = c1 + 0.5*c3, y3c*c3
        # Origin RGB values
        rgbVals = np.dstack((c1, c2, c3))
        # For better barycentric map, use transformation on c1, c2, c3, as in Emory et al. (2014)
        # ci_star = (ci + c_offset)^c_exp,
        # Improved RGB = [c1_star, c2_star, c3_star]
        rgbValsNew = np.empty((c1.shape[0], c1.shape[1], 3))
        # Each 3rd dim is an RGB array of the 2D grid
        for i in range(3):
            rgbValsNew[:, :, i] = (rgbVals[:, :, i] + c_offset)**c_exp

        return xBary, yBary, rgbValsNew


    @staticmethod
    @timer
    def mergeHorizontalComponents(valsDecomp):
        valsDecomp['hor'] = np.sqrt(valsDecomp['0']**2 + valsDecomp['1']**2)
        return valsDecomp


    @staticmethod
    @timer
    def calcSliceMeanDissipationRate(epsilonSGSmean, nuSGSmean, nu=1e-5, save=False, result_path='.'):
        # FIXME: DEPRECATED
        # According to Eq 5.64 - Eq 5.68 of Sagaut (2006), for isotropic homogeneous turbulence,
        # <epsilon> = <epsilon_resolved> + <epsilon_SGS>,
        # <epsilon_resolved>/<epsilon_SGS> = 1/(1 + <nu_SGS>/nu),
        # where epsilon is the total turbulence dissipation rate (m^2/s^3); and <> is statistical averaging

        epsilonMean = epsilonSGSmean/(1. - (1./(1. + nuSGSmean/nu)))
        # Avoid FPE
        epsilonMean[epsilonMean == np.inf] = 1e10
        epsilonMean[epsilonMean == -np.inf] = -1e10

        if save:
            pickle.dump(epsilonMean, open(result_path + '/epsilonMean.p', 'wb'))
            print('\nepsilonMean saved at {0}'.format(result_path))

        return epsilonMean


    @staticmethod
    @timer
    def calcSliceSijAndRij(grad_u, tke, eps, cap = 7.):
        """
        Calculates the non-dimonsionalized strain rate and rotation rate tensors.  Normalizes by k and eps:
        Sij = k/eps * 0.5* (grad_u  + grad_u^T)
        Rij = k/eps * 0.5* (grad_u  - grad_u^T)
        :param grad_u: num_points X 3 X 3
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate epsilon
        :param cap: This is the max magnitude that Sij or Rij components are allowed.  Greater values
                    are capped at this level
        :return: Sij, Rij: num_points X 3 X 3 tensors
        """

        if len(grad_u.shape) == 2 and grad_u.shape[1] == 9:
            grad_u = grad_u.reshape((grad_u.shape[0], 3, 3))

        if len(tke.shape) == 2:
            tke = tke.ravel()

        if len(eps.shape) == 2:
            eps = eps.ravel()

        num_points = grad_u.shape[0]
        eps = np.maximum(eps, 1e-8)
        tke_eps = tke/eps
        Sij = np.zeros((num_points, 3, 3))
        Rij = np.zeros((num_points, 3, 3))
        for i in range(num_points):
            Sij[i, :, :] = tke_eps[i]*0.5*(grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            Rij[i, :, :] = tke_eps[i]*0.5*(grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        maxSij, maxRij = np.amax(Sij), np.amax(Rij)
        minSij, minRij = np.amin(Sij), np.amin(Rij)
        print(' Max of Sij is ' + str(maxSij) + ', and of Rij is ' + str(maxRij))
        print(' Min of Sij is ' + str(minSij) + ', and of Rij is ' + str(minRij))
        # Why caps?????????????
        Sij[Sij > cap] = cap
        Sij[Sij < -cap] = -cap
        Rij[Rij > cap] = cap
        Rij[Rij < -cap] = -cap

        # Because we enforced limits on maximum Sij values, we need to re-enforce trace of 0
        for i in range(num_points):
            Sij[i, :, :] = Sij[i, :, :] - 1/3.*np.eye(3)*np.trace(Sij[i, :, :])

        return Sij, Rij
    
    
    @staticmethod
    @timer
    def calcSliceScalarBasis(Sij, Rij, is_train = False, cap = 2.0, is_scale = True, mean = None, std = None):
        """
        Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
        this returns a set of normalized scalar invariants
        :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
        :param Rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        :return: mean: Mean of the scalar basis, can be used when predicting
        :return: std: Standard deviation of the scalar basis, can be used when predicting
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3) * 2.0
        >>> B[0, 1, 0] = 1.0
        >>> B[0, 0, 1] = -1.0
        >>> tdp = PostProcess_SliceData()
        >>> scalar_basis, mean, std = tdp.calcSliceScalarBasis(A, B, is_scale=False)
        >>> print scalar_basis
        [[ 12.  -2.  24.  -4.  -8.]]
        """
        if is_train or mean is None or std is None:
            print("Re-setting normalization constants")
            
        num_points = Sij.shape[0]
        num_invariants = 5
        invariants = np.zeros((num_points, num_invariants))
        for i in range(num_points):
            invariants[i, 0] = np.trace(np.dot(Sij[i, :, :], Sij[i, :, :]))
            invariants[i, 1] = np.trace(np.dot(Rij[i, :, :], Rij[i, :, :]))
            invariants[i, 2] = np.trace(np.dot(Sij[i, :, :], np.dot(Sij[i, :, :], Sij[i, :, :])))
            invariants[i, 3] = np.trace(np.dot(Rij[i, :, :], np.dot(Rij[i, :, :], Sij[i, :, :])))
            invariants[i, 4] = np.trace(
                np.dot(np.dot(Rij[i, :, :], Rij[i, :, :]), np.dot(Sij[i, :, :], Sij[i, :, :])))

        # Renormalize invariants using mean and standard deviation:
        if is_scale:
            if mean is None or std is None:
                is_train = True

            if is_train:
                mean = np.zeros((num_invariants, 2))
                std = np.zeros((num_invariants, 2))
                mean[:, 0] = np.mean(invariants, axis = 0)
                std[:, 0] = np.std(invariants, axis = 0)

            invariants = (invariants - mean[:, 0])/std[:, 0]
            maxInvariants, minInvariants = np.amax(invariants), np.amin(invariants)
            print(' Max of scaled scalar basis is {}'.format(maxInvariants))
            print(' Min of scaled scalar basis is {}'.format(minInvariants))
            # Why cap?????
            invariants[invariants > cap] = cap  # Cap max magnitude
            invariants[invariants < -cap] = -cap
            invariants = invariants*std[:, 0] + mean[:, 0]
            if is_train:
                mean[:, 1] = np.mean(invariants, axis = 0)
                std[:, 1] = np.std(invariants, axis = 0)

            invariants = (invariants - mean[:, 1])/std[:, 1]  # Renormalize a second time after capping
        return invariants, mean, std


    @staticmethod
    @timer
    @njit(parallel=True, fastmath=True)
    def calcSliceTensorBasis(Sij, Rij, quadratic_only = False, is_scale = True):
        """
        Given non-dimsionalized Sij and Rij, it calculates the tensor basis
        :param Sij: normalized strain rate tensor
        :param Rij: normalized rotation rate tensor
        :param quadratic_only: True if only linear and quadratic terms are desired.  False if full basis is desired.
        :return: T_flat: num_points X num_tensor_basis X 9 numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3)
        >>> B[0, 1, 0] = 3.0
        >>> B[0, 0, 1] = -3.0
        >>> tdp = PostProcess_SliceData()
        >>> tb = tdp.calcSliceTensorBasis(A, B, is_scale=False)
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
        num_tensor_basis = 10 if not quadratic_only else 4
        T = np.zeros((Sij.shape[0], num_tensor_basis, 3, 3))
        for i in range(Sij.shape[0]):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            T[i, 0, :, :] = sij
            T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
            T[i, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
            T[i, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
            if not quadratic_only:
                T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
                T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) \
                                + np.dot(sij, np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
                T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
                T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
                T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) \
                                + np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
                T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                                - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
            # Enforce zero trace for anisotropy
            for j in range(num_tensor_basis):
                T[i, j, :, :] = T[i, j, :, :] - 1./3.*np.eye(3)*np.trace(T[i, j, :, :])

        # Scale down to promote convergence
        if is_scale:
            # Using tuple here gievs Numbe error
            scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
            for i in range(num_tensor_basis):
                T[:, i, :, :] /= scale_factor[i]

        # Flatten:
        # T_flat = np.zeros((num_points, num_tensor_basis, 9))
        # for i in range(3):
        #     for j in range(3):
        #         T_flat[:, :, 3*i+j] = T[:, :, i, j]
        T_flat = T.reshape((T.shape[0], T.shape[1], 9))

        return T_flat


    @staticmethod
    @timer
    def rotateSpatialCorrelationTensors(listData, rotateXY=0., rotateUnit='rad', dependencies=('xx',)):
        """
        Rotate one or more single/double spatial correlation scalar/tensor field/slice data in the x-y plane,
        doesn't work on rate of strain/rotation tensors
        :param listData: Any (nPt x nComponent) or (nX x nY x nComponent) or (nX x nY x nZ x nComponent) data of interest, appended to a tuple/list.
        If nComponent is 6, data is symmetric 3 x 3 double spatial correlation tensor field.
        If nComponent is 9, data is single/double spatial correlation tensor field depending on dependencies keyword
        :type listData: tuple/list([:, :]/[:, :, :]/[:, :, :, :])
        :param rotateXY:
        :type rotateXY:
        :param rotateUnit:
        :type rotateUnit:
        :param dependencies: Whether the component of data is dependent on single spatial correlation 'x' e.g. gradient,vector,
        or double spatial correlation 'xx' e.g. double correlation.
        Only used if nComponent is 9
        :type dependencies: Str or list/tuple of 'x' or 'xx'. Default is ('xx',)
        :return: listData_rot
        :rtype:
        """
        # Ensure list input since each listData[i] is modified to nPt x nComponent later
        listData = list((listData,)) if isinstance(listData, np.ndarray) else list(listData)
        # Ensure tuple input
        dependencies = (dependencies,) if isinstance(dependencies, str) else dependencies
        # Ensure dependencies has the same entries as the number of data provided
        dependencies *= len(listData) if len(dependencies) < len(listData) else 1
        # Ensure radian unit
        rotateXY *= np.pi/180 if rotateUnit != 'rad' else 1.
        # Reshape here screwed up njit :/
        def __transform(listData, rotateXY, dependencies):
            # Copy listData (a list) that has original shapes as listData will be flattened to nPt x nComponent
            listDataRot_oldShapes = listData.copy()
            sinVal, cosVal = np.sin(rotateXY), np.cos(rotateXY)
            # Go through every data in listData and flatten to nPt x nComponent if necessary
            for i in range(len(listData)):
                # # Ensure Numpy array
                # listData[i] = np.array(listData[i])
                # Flatten data from 3D to 2D
                if len(listData[i].shape) >= 3:
                    # Go through 2nd D to (last - 1) D
                    nRow = 1
                    for j in range(len(listData[i].shape) - 1):
                        nRow *= listData[i].shape[j]

                    # Flatten data to nPt x nComponent
                    nComponent = listData[i].shape[len(listData[i].shape) - 1]
                    listData[i] = listData[i].reshape((nRow, nComponent))

            # Create a copy so listData values remain unchanged during the transformation, listData[i] is nPt x nComponent now
            listData_rot = listData.copy()
            # Go through all provided data and perform transformation
            for i in range(len(listData)):
                # Number of component is the last D
                nComponent = listData[i].shape[len(listData[i].shape) - 1]
                # If nComponent is 3, i.e. x, y, z or data is single spatial correlation with 9 components
                if nComponent == 3 or (nComponent == 9 and dependencies[i] == 'x'):
                    # x_rot becomes x*cos + y*sin
                    x_rot = listData[i][:, 0]*cosVal + listData[i][:,
                                                       1]*sinVal
                    # y_rot becomes -x*sin + y*cos
                    y_rot = -listData[i][:, 0]*sinVal + listData[i][:,
                                                        1]*cosVal
                    # z_rot doesn't change
                    z_rot = listData[i][:, 2]
                    listData_rot[i][:, 0], listData_rot[i][:, 1], listData_rot[i][:, 2] = x_rot, y_rot, z_rot
                    # If 9 components with single spatial correlation, e.g.gradient tensor, do it for 2nd row and 3rd row
                    if nComponent == 9:
                        x_rot2 = listData[i][:, 3]*cosVal + listData[i][:,
                                                            4]*sinVal
                        y_rot2 = -listData[i][:, 3]*sinVal + listData[i][:,
                                                            4]*cosVal
                        z_rot2 = listData[i][:, 5]
                        x_rot3 = listData[i][:, 6]*cosVal + listData[i][:,
                                                            7]*sinVal
                        y_rot3 = -listData[i][:, 6]*sinVal + listData[i][:,
                                                             7]*cosVal
                        z_rot3 = listData[i][:, 8]
                        listData_rot[i][:, 3], listData_rot[i][:, 4], listData_rot[i][:, 5] = x_rot2, y_rot2, z_rot2
                        listData_rot[i][:, 6], listData_rot[i][:, 7], listData_rot[i][:, 8] = x_rot3, y_rot3, z_rot3

                # Else if nComponent is 6 or 9 with double spatial correlation
                elif nComponent == 6 or (nComponent == 9 and dependencies[i] == 'xx'):
                    # If 6 components and double spatial correlation, i.e.xx, xy, xz, yy, yz, zz
                    # or 9 components and double spatial correlation, i.e. xx, xy, xz, yx, yy, yz, zx, zy, zz
                    if dependencies[i] == 'xx':
                        xx, xy, xz = listData[i][:, 0], listData[i][:, 1], listData[i][:, 2]
                        yy, yz = listData[i][:, 3], listData[i][:, 4]
                        zz = listData[i][:, 5]
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

        return __transform(listData, rotateXY, rotateUnit, dependencies)


    def processSliceTBNN_Inputs(self, gradUAvg, kMean, epsilonMean, capSijRij = 1e9, capSB = 1e9, scaleSB = True, scaleTB = True):
        # Calculate the non-dimensionalized Sij and Rij using kMean/epsilonMean
        Sij, Rij = self.calcSliceSijAndRij(gradUAvg, kMean, epsilonMean, capSijRij)
        # Calculate the 5 scalar basis using Sij and Rij
        scalarBasis, mean, std = self.calcSliceScalarBasis(Sij, Rij, is_train = True, is_scale = scaleSB, cap = capSB)
        # Calculate the 10 tensor basis using Sij and Rij
        tensorBasis = self.calcSliceTensorBasis(Sij, Rij, scaleTB)

        return scalarBasis, tensorBasis, mean, std












if __name__ is '__main__':
    import time as t
    from PlottingTool import PlotSurfaceSlices3D
    import matplotlib.pyplot as plt
    from scipy import ndimage
    import pickle

    """
    User Inputs
    """
    time = 'latest'
    casedir = 'J:'
    casedir = '/media/yluan/1'
    casename = 'ALM_N_H_ParTurb'
    property = 'uuPrime2'
    slicenames = 'alongWindRotorOne'
    # Orientation of x-axis in x-y plane, in case of angled flow direction
    # Only used for values decomposition
    # Angle in rad and counter-clockwise
    rot_z = 6/np.pi
    precision_x, precision_y, precision_z = 1000j, 1000j, 333j
    interp_method = 'nearest'
    plot = 'bary'

    case = SliceProperties(time = time, casedir = casedir, casename = casename, rot_z = rot_z)

    case.readSlices(property = property, slicenames = slicenames)

    for slicename in case.slicenames:
        """
        Process Uninterpolated Anisotropy Tensor
        """
        vals2D = case.slices_val[slicename]
        # x2d, y2d, z2d, vals3d = \
        #     case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0], case.slices_coor[slicename][:, 1], case.slices_coor[slicename][:, 2], case.slices_val[slicename], slice_orient = case.slice_orient[slicename], rot_z = case.rot_z, precision_x = precision_x, precision_y = precision_y, precision_z = precision_z, interp_method = interp_method)

        # Another implementation of processAnisotropyTensor() in Cython
        t0 = t.time()
        # vals3d, tensors, eigValsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor(vals3d)
        # In each eigenvector 3 x 3 matrix, 1 col is a vector
        vals2D, tensors, eigValsGrid, eigVecsGrid = PostProcess_AnisotropyTensor.processAnisotropyTensor_Uninterpolated(vals2D, realizeIter = 0)
        t1 = t.time()
        ticToc = t1 - t0

        # valsDecomp, tensors, eigValsGrid = case.processAnisotropyTensor_Fast(valsDecomp)
        # vals2D, tensors, eigValsGrid = case.processAnisotropyTensor_Uninterpolated(vals2D)


        """
        Interpolation
        """
        if plot == 'bary':
            xBary, yBary, rgbVals = case.getBarycentricMapCoordinates(eigValsGrid)

            x2d, y2d, z2d, rgbVals = \
                case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0], case.slices_coor[slicename][:, 1], case.slices_coor[slicename][:, 2], rgbVals, slice_orient =
                                                         case.slice_orient[slicename], rot_z = case.rot_z,
                                                         precision_x = precision_x, precision_y =
                                                         precision_y, precision_z = precision_z, interp_method = interp_method)

        elif plot == 'quiver':
            x2d, y2d, z2d, eigVecs3D = \
                case.interpolateDecomposedSliceData_Fast(case.slices_coor[slicename][:, 0], case.slices_coor[slicename][:, 1],
                                                         case.slices_coor[slicename][:, 2], eigVecsGrid[:, :, :, 0], slice_orient =
                                                         case.slice_orient[slicename], rot_z = case.rot_z,
                                                         precision_x = precision_x, precision_y =
                                                         precision_y, precision_z = precision_z, interp_method = interp_method)


        """
        Plotting
        """
        if plot == 'bary':
            print('\nDumping values...')
            pickle.dump(tensors, open(case.result_path + slicename + '_rgbVals.p', 'wb'))
            pickle.dump(x2d, open(case.result_path + slicename + '_x2D.p', 'wb'))
            pickle.dump(y2d, open(case.result_path + slicename + '_y2D.p', 'wb'))
            pickle.dump(z2d, open(case.result_path + slicename + '_z2D.p', 'wb'))

            # # Custom RGB colormap, with only red, green, and blue
            # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            # nBin = 1000
            # cmapName = 'rgb'
            # from matplotlib.colors import LinearSegmentedColormap
            # rgbCm = LinearSegmentedColormap.from_list(
            #         cmapName, colors, N = nBin)

            # rgbVals[rgbVals < 0] = 0.
            # rgbVals[rgbVals > 1] = 1.

            # Rotate the figure 90 deg clockwise
            rgbValsRot = ndimage.rotate(rgbVals, 90)
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(rgbValsRot, origin = 'upper', aspect = 'equal', extent = (0, 3000, 0, 3000))
            # fig.colorbar(im, ax = ax, extend = 'both')
            plt.savefig('R:/bary.png', dpi = 1000)

            # baryPlot = PlotSurfaceSlices3D(x2d, y2d, z2d, (0,), show = True, name = 'bary', figDir = 'R:', save = True)
            # baryPlot.cmapLim = (0, 1)
            # baryPlot.cmapNorm = rgbVals
            # # baryPlot.cmapVals = plt.cm.ScalarMappable(norm = rgbVals, cmap = None)
            # baryPlot.cmapVals = rgbVals
            # # baryPlot.cmapVals.set_array([])
            # baryPlot.plot = baryPlot.cmapVals
            # baryPlot.initializeFigure()
            # baryPlot.axes[0].plot_surface(x2d, y2d, z2d, cstride = 1, rstride = 1, facecolors = rgbVals, vmin = 0, vmax = 1, shade = False)
            # baryPlot.finalizeFigure()
            #
            #
            #
            #
            #
            # print('\nDumping values...')
            # pickle.dump(tensors, open(case.result_path + slicename + '_tensors.p', 'wb'))
            # pickle.dump(case.slices_coor[slicename][:, 0], open(case.result_path + slicename + '_x.p', 'wb'))
            # pickle.dump(case.slices_coor[slicename][:, 1], open(case.result_path + slicename + '_y.p', 'wb'))
            # pickle.dump(case.slices_coor[slicename][:, 2], open(case.result_path + slicename + '_z.p', 'wb'))
            #
            # print('\nExecuting RGB_barycentric_colors_clean...')
            # import RBG_barycentric_colors_clean
            #
            #
            #
            #
            # # valsDecomp = case.mergeHorizontalComponents(valsDecomp)
        elif plot == 'quiver':
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            ax.quiver(x2d, y2d, z2d, eigVecs3D[:, :, 0], eigVecs3D[:, :, 1], eigVecs3D[:, :, 2], length = 0.1, normalize = False)

