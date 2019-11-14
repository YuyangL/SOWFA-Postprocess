import os
import numpy as np
from warnings import warn
from Utility import timer
import shutil

class BaseProperties:
    def __init__(self, casename, casedir='.', filename_pre='', filename_sub='', ensemblefolder_name='Ensemble', result_folder='Result', timecols='infer', time_kw='ime', force_remerge=False,
                 debug=False, **kwargs):
        self.casename, self.casedir = casename, casedir
        self.case_fullpath = casedir + '/' + casename + '/'
        self.filename_pre, self.filename_sub = filename_pre, filename_sub
        # Check if result folder is made
        self.result_dir = self.case_fullpath + result_folder + '/'
        os.makedirs(self.result_dir, exist_ok=True)
        self.ensemble_folderpath = self.case_fullpath + ensemblefolder_name + '/'
        self.timecols = (timecols,) if isinstance(timecols, int) else timecols
        self._mergeTimeDirectories(time_kw=time_kw, force_remerge=force_remerge, **kwargs)
        self.times_all = self._readTimes(**kwargs)
        self.data, self.filenames = {}, []
        self.debug = debug

    def _ensureTupleInput(self, input):
        input_tuple = (input,) if isinstance(input, (str, np.ndarray, int)) else input
        # If input[0] is '*' or 'all', get all file names
        # input_tuple = os.listdir(self.ensemble_folderpath) if input_tuple[0] in ('*', 'all') else input_tuple
        if input_tuple[0] in ('*', 'all'):
            # Ignore hidden files
            input_tuple = tuple([f for f in os.listdir(self.ensemble_folderpath) if not f.startswith('.')])

        return input_tuple

    def _readTimes(self, norepeat=True, **kwargs):
        filenames = os.listdir(self.ensemble_folderpath)
        # In case of file e.g. hLevelsCell that doesn't incl. times
        try:
            times_all = np.genfromtxt(self.ensemble_folderpath + '/' + filenames[0])[:, self.timecols[0]]
        except IndexError:
            times_all = np.genfromtxt(self.ensemble_folderpath + '/' + filenames[1])[:, self.timecols[0]]
            
        if norepeat:
            times_all = np.unique(times_all)

        return times_all

    def _mergeTimeDirectories(self, trim_overlaptime=True, time_kw='ime', force_remerge=False, excl_files=None):
        def __numberStringToFloat(str):
            return float(str)

        # Check whether the directory is made
        try:
            os.makedirs(self.ensemble_folderpath)
        except OSError:
            # If folder not empty, abort
            if os.listdir(self.ensemble_folderpath) and not force_remerge:
                print('\n{0} files already exist'.format(self.ensemble_folderpath))
                return

        # Available time directories (excluding Ensemble and Result) and file names
        timedirs = os.listdir(self.case_fullpath)[:-2]
        # Sort the time directories
        timedirs.sort(key=__numberStringToFloat)
        filenames = os.listdir(self.case_fullpath + timedirs[0])
        # In case excl_files is provided, remove it from filenames
        if excl_files is not None:
            excl_files = self._ensureTupleInput(excl_files)
            for i in range(len(excl_files)):
                try:
                    filenames.remove(excl_files[i])
                except ValueError:
                    warn('\n' + self.casename + ' does not have ' + excl_files[i] + ' to exclude!', stacklevel=2)
                    pass

        # Initialize ensemble files
        file_ensembles = {}
        for i in range(len(filenames)):
            file_ensembles[filenames[i]] = open(self.ensemble_folderpath + filenames[i], "w")

        if self.timecols == 'infer':
            self.timecols = []
            for filename in filenames:
                with open(self.case_fullpath + timedirs[0] + '/' + filename, 'r') as file:
                    header = (file.readline()).split()
                    self.timecols.append(header.index(list(filter(lambda kw: time_kw in kw, header))[0]))
        else:
            self.timecols *= len(filenames)

        # Go through time folders and append files to ensemble
        # Excluding Ensemble folder
        for i in range(len(timedirs)):
            # If trim overlapped time and not in last time directory
            if trim_overlaptime and i < len(timedirs) - 1:
                knowntime_cols, times, itrim = {}, {}, {}
                # Go through all time columns of each file in order
                for j in range(len(self.timecols)):
                    # Retrieve list of time and trim index information for jth file in ith time directory
                    # After each retrieval, add this time column to known time column dictionary as key
                    # and corresponding file name as value
                    if str(self.timecols[j]) not in knowntime_cols.keys():
                        try:
                            times[filenames[j]] = np.genfromtxt(self.case_fullpath + timedirs[i] + '/' + filenames[j])[:, self.timecols[j]]
                        # In case the last line wasn't written properly,
                        # which means the simulation was probably aborted, discard the last line
                        except ValueError:
                            times[filenames[j]] = np.genfromtxt(self.case_fullpath + timedirs[i] + '/' + filenames[j], skip_footer = 1)[:, self.timecols[j]]

                        # Index at which trim should start for this file
                        itrim[filenames[j]] = np.searchsorted(times[filenames[j]], np.float_(timedirs[i + 1]))
                        # Add this time column to known time column list
                        knowntime_cols[str(self.timecols[j])] = filenames[j]

                    # If current time column already exists in remembered dictionary,
                    # then skip it and retrieve the file name the last time it had this number of time column
                    else:
                        times[filenames[j]] = times[knowntime_cols[str(self.timecols[j])]]
                        itrim[filenames[j]] = itrim[knowntime_cols[str(self.timecols[j])]]

            # Go through each file in this time directory
            for filename in filenames:
                # If trim overlapped time and not last time directory and trim is indeed needed
                if trim_overlaptime and i < len(timedirs) - 1 and itrim[filename] < (len(times[filename]) - 1):
                    with open(self.case_fullpath + timedirs[i] + '/' + filename, 'r') as file:
                        # Filter out empty lines before itrim indices can be mapped
                        lines = list(filter(None, (line.rstrip() for line in file)))
                        
                    print('\nTrimming overlapped time and adding {0} from {1} to Ensemble...'.format(filename, timedirs[i]))
                    # Writelines support writing a 1D list, since lines is 2D,
                    # join each row with "\n"
                    # Note: the header of 2nd file onward will still be written in ensemble,
                    # just that when reading file into array using numpy, the headers should automatically be ignored
                    # since it starts with "#"
                    # Write the 1st line as empty new line so that the 1st line of lines is not on the same line as last line of file_ensembles
                    file_ensembles[filename].writelines("\n")
                    file_ensembles[filename].writelines("\n".join(lines[:itrim[filename] + 1]))
                # Otherwise, append this file directly to Ensemble
                else:
                    print('\nAdding {0} from {1} to Ensemble...'.format(filename, timedirs[i]))
                    # Again, write the 1st line as empty new line to avoid 1st line of next file being on the same line of old file
                    file_ensembles[filename].writelines("\n")
                    file_ensembles[filename].write(open(self.case_fullpath + timedirs[i] + '/' + filename).read())

        print("\nMerged time directories for " + str(self.casename) + " files are stored at:\n " + str(self.ensemble_folderpath))

    def _selectTimes(self, starttime=None, stoptime=None):
        starttime = self.times_all[0] if starttime is None else starttime
        stoptime = self.times_all[len(self.times_all)] if stoptime is None else stoptime
        # Bisection left to find actual starting and ending time and their indices
        (istart, istop) = np.searchsorted(self.times_all, (starttime, stoptime))
        # If stoptime larger than any time, istop = len(times_all)
        istop = min(istop, len(self.times_all) - 1)
        starttime_real, stoptime_real = self.times_all[istart], self.times_all[istop]
        times_selected = self.times_all[istart:istop + 1]

        return times_selected, starttime_real, stoptime_real, istart, istop

    def readPropertyData(self, filenames=('*',), skiprow=0, skipcol=0, skipfooter=0):
        self.filenames = self._ensureTupleInput(filenames)
        if isinstance(skiprow, int): skiprow = (skiprow,)*len(self.filenames)
        if isinstance(skipcol, int): skipcol = (skipcol,)*len(self.filenames)
        if isinstance(skipfooter, int): skipfooter = (skipfooter,)*len(self.filenames)

        for i in range(len(self.filenames)):
            # Data dictionary of specified property(s) of all times
            self.data[self.filenames[i]] = \
                np.genfromtxt(self.ensemble_folderpath + self.filename_pre + self.filenames[i] + self.filename_sub, 
                              skip_footer=skipfooter[i])[skiprow[i]:, skipcol[i]:]

        print('\n' + str(self.filenames) + ' read')

    def calculatePropertyMean(self, axis=1, starttime=None, stoptime=None):
        self.times_selected, _, _, istart, istop = self._selectTimes(starttime=starttime, stoptime=stoptime)
        for i in range(len(self.filenames)):
            self.data[self.filenames[i] + '_mean'] = np.mean(self.data[self.filenames[i]][istart:istop], 
                                                             axis=axis)
        
        print('\nTemporal average calculated for {} from {:.4f} s - {:.4f} s'.format(self.filenames, self.times_selected[0], self.times_selected[-1]))

    def trimInvalidCharacters(self, filenames, invalid_chars):
        filenames = self._ensureTupleInput(filenames)
        if isinstance(invalid_chars, str): invalid_chars = (invalid_chars,)
        for filename in filenames:
            with open(self.ensemble_folderpath + filename, 'r') as f:
                lst = [line.rstrip('\n \t') for line in f]

            for invalid_char in invalid_chars:
                lst = [string.replace(invalid_char, '') for string in lst]

            with open(self.ensemble_folderpath + filename, "w") as f:
                f.writelines('\n'.join(lst))


class BoundaryLayerProfiles(BaseProperties):
    def __init__(self, casename, height_filename='hLevelsCell', bl_folder='Inflows', **kwargs):
        self.height_filename = height_filename
        super(BoundaryLayerProfiles, self).__init__(casename=casename + '/' + bl_folder, timecols=0, excl_files=height_filename, **kwargs)
        # Copy height_filename to Ensemble in order to use it later
        time = os.listdir(self.case_fullpath)[0]
        shutil.copy2(self.case_fullpath + time + '/' + height_filename, self.ensemble_folderpath)

    def readPropertyData(self, filenames=('*',), **kwargs):
        # Read height levels
        self.hLvls = np.genfromtxt(self.ensemble_folderpath + self.height_filename)
        # Override skipcol to suit inflow property files
        # Columns to skip are 0: time; 1: time step
        super(BoundaryLayerProfiles, self).readPropertyData(filenames=filenames, skipcol=2)


    def calculatePropertyMean(self, starttime=None, stoptime=None, **kwargs):
        # Override axis to suit inflow property files
        super(BoundaryLayerProfiles, self).calculatePropertyMean(axis = 0, starttime = starttime, stoptime = stoptime, **kwargs)


class TurbineOutputs(BaseProperties):
    def __init__(self, casename, datafolder='turbineOutput', global_quantities=('powerRotor', 'rotSpeed', 'thrust',
                                                                      'torqueRotor',
                                                       'torqueGen', 'azimuth', 'nacYaw', 'pitch'), **kwargs):
        self.global_quantities = global_quantities
        super(TurbineOutputs, self).__init__(casename + '/' + datafolder, **kwargs)

        self.nTurb, self.nBlade = 0, 0

    @timer
    def readPropertyData(self, filenames=('*',), skiprow=0, skipcol='infer', verbose=True, turbinfo=('infer',)):
        filenames = self._ensureTupleInput(filenames)
        global_quantities = (
        'powerRotor', 'rotSpeed', 'thrust', 'torqueRotor', 'torqueGen', 'azimuth', 'nacYaw', 'pitch', 'powerGenerator')
        if skipcol is 'infer':
            skipcol = []
            for file in filenames:
                if file in global_quantities:
                    skipcol.append(3)
                else:
                    skipcol.append(4)

        super(TurbineOutputs, self).readPropertyData(filenames=filenames, skiprow=skiprow, skipcol=skipcol)

        if turbinfo[0] is 'infer':
            turbinfo = np.genfromtxt(self.ensemble_folderpath + self.filename_pre + 'Cl' + self.filename_sub)[skiprow:, :2]

        # Number of turbines and blades
        (self.nTurb, self.nBlade) = (int(np.max(turbinfo[:, 0]) + 1), int(np.max(turbinfo[:, 1]) + 1))

        fileNamesOld, self.filenames = self.filenames, list(self.filenames)
        for filename in fileNamesOld:
            for i in range(self.nTurb):
                if filename not in global_quantities:
                    for j in range(self.nBlade):
                        newFileName = filename + '_Turb' + str(i) + '_Bld' + str(j)
                        self.data[newFileName] = self.data[filename][(i*self.nBlade + j)::(self.nTurb*self.nBlade), :]
                        self.filenames.append(newFileName)

                else:
                    newFileName = filename + '_Turb' + str(i)
                    self.data[newFileName] = self.data[filename][i::self.nTurb]
                    self.filenames.append(newFileName)

        if verbose:
            print('\n' + str(self.filenames) + ' read')


class InflowBoundaryField(BaseProperties):
    def __init__(self, casename, casedir='.', boundarydata_folder='boundaryData', avg_folder='Average', debug=False, **kwargs):
        self.casename, self.casedir = casename, casedir
        self.case_fullpath = casedir + '/' + casename + '/' + boundarydata_folder + '/'
        self.inflow_patches = os.listdir(self.case_fullpath)
        # Try remove "Average" folder from collected patch names
        try:
            self.inflow_patches.remove(avg_folder)
        except ValueError:
            pass

        self.avg_folder_path = self.case_fullpath + avg_folder + '/'
        # Patch folder paths in Average folder
        self.avg_folder_patchpaths, self.case_patchfullpath = [], []
        for patch in self.inflow_patches:
            self.avg_folder_patchpaths.append(self.avg_folder_path + patch + '/')
            self.case_patchfullpath.append(self.case_fullpath + patch + '/')
            # Try making Average folder and its subfolder, if not already
            os.makedirs(self.avg_folder_path + patch + '/', exist_ok=True)

        self.data, self.data_mean = {}, {}
        # Exception for inheritance class DrivingPressureGradient
        try:
            self.times_all, self.times_all_raw = self._readTimes(**kwargs)
        except NotADirectoryError:
            pass

        self.debug = debug
        print('{} InflowBoundaryField object initialized'.format(casename))

    def _readTimes(self, remove='points', **kwargs):
        times_all = os.listdir(self.case_patchfullpath[0])
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

    @staticmethod
    def _trimBracketCharacters(data):
        # Get left and right column of U
        datacol0, datacol1, datacol2 = data['f0'], data['f1'], data['f2']
        # New corrected data
        data_new = np.empty((data.shape[0], 3, data.shape[2]))
        # Go through each point then each time
        for l in range(data.shape[0]):
            # print(l)
            for m in range(data.shape[2]):
                newval0, newval2 = datacol0[l, 0, m].decode('utf-8'), datacol2[l, 0, m].decode('utf-8')
                data_new[l, 0, m] = float(newval0.lstrip('('))
                data_new[l, 1, m] = datacol1[l, 0, m]
                # Right column doesn't need to strip ) since precision limit was 10 and not enough to reach ")"
                data_new[l, 2, m] = float(newval2.rstrip(')'))

        return data_new

    @timer
    def readPropertyData(self, filenames=('*',), skiprow=22, skipfooter=1, n_timesample=-1, lstr_precision=12, rstr_precision=20):
        # Ensure tuple inputs and interpret "*" as all files
        # ensemble_folderpath is a dummy variable here
        self.ensemble_folderpath = self.case_patchfullpath[0] + self.times_all_raw[0] + '/'
        self.filenames = self._ensureTupleInput(filenames)
        self.ensemble_folderpath = ''
        # Ensure same size as number of files specified
        skiprow = (skiprow,)*len(self.filenames) if isinstance(skiprow, int) else skiprow
        skipfooter = (skipfooter,)*len(self.filenames) if isinstance(skipfooter, int) else skipfooter
        # If n_timesample is -1 or sample interval < 1.5, then use all times
        sample_interval = 1 if n_timesample == -1 or n_timesample > len(self.times_all)/1.5 else int(np.ceil(len(self.times_all))/n_timesample)
        self.sample_times = [self.times_all[0]] if sample_interval > 1 else self.times_all
        # Go through all specified properties
        for i in range(len(self.filenames)):
            # String dtype for left and right column of U so that "(12345" can be read, precision is lstr_precision and rstr_precision
            dtype = ('|S' + str(lstr_precision), float, '|S' + str(rstr_precision)) if self.filenames[i] == 'U' else float
            # Go through all patches
            for j in range(len(self.inflow_patches)):
                print('\nReading {}'.format(self.filenames[i] + ' ' + self.inflow_patches[j]))
                filename_fullpath = self.case_patchfullpath[j] + self.times_all_raw[0] + '/' + self.filenames[i]
                property_dictkey = self.filenames[i] + '_' + self.inflow_patches[j]
                # Initialize first index in the 3rd dimension
                data = np.genfromtxt(filename_fullpath, skip_header=skiprow[i], skip_footer=skipfooter[i], dtype=dtype)
                # Then go through all times from 2nd time onward
                cnt, milestone = 0, 25
                for k in range(sample_interval, len(self.times_all), sample_interval):
                    # print(self.filenames[i] + ' ' + self.inflow_patches[j] + ' ' + str(self.times_all[k]))
                    filename_fullpath = self.case_patchfullpath[j] + self.times_all_raw[k] + '/' + self.filenames[i]
                    data_pertime = np.genfromtxt(filename_fullpath, skip_header=skiprow[i], skip_footer=skipfooter[i], dtype=dtype)
                    data = np.dstack((data, data_pertime))
                    # Gauge progress
                    cnt += sample_interval
                    progress = cnt/(len(self.times_all) + 1)*100.
                    if progress >= milestone:
                        print(' ' + str(milestone) + '%...', end='')
                        milestone += 25

                # Some postprocessing after reading and dstacking data per time
                data = data.reshape((data.shape[1], data.shape[0], data.shape[2]))
                # If file is U, then strip "(" and ")"
                if self.filenames[i] == 'U':
                    data_new = self._trimBracketCharacters(data)
                else:
                    data_new = data

                # Finally, the property data
                self.data[property_dictkey] = data_new

        # Collect sample times
        self.sample_times = np.empty(data_new.shape[2])
        i = 0
        for k in range(0, len(self.times_all), sample_interval):
            self.sample_times[i] = self.times_all[k]
            i += 1

        # Collect all property keys
        self.property_keys = tuple(self.data.keys())
        # Numpy array treatment
        self.sample_times = np.array(self.sample_times)
        print('\n' + str(self.filenames) + ' read')

    def _readPoints(self, points_name='points', skiprow=18, skipfooter=1, lstr_precision=12, rstr_precision=20):
        self.points_name = points_name
        dtype = ('|S' + str(lstr_precision), float, '|S' + str(rstr_precision))
        self.points = {}
        # Go through all patches
        for i in range(len(self.inflow_patches)):
            print('\nReading {}'.format(self.points_name + ' ' + self.inflow_patches[i]))
            filename_fullpath = self.case_patchfullpath[i] + '/' + self.points_name
            points_dictkey = self.points_name + '_' + self.inflow_patches[i]
            points = np.genfromtxt(filename_fullpath, skip_header=skiprow, skip_footer=skipfooter, dtype=dtype)
            # Strip "(" and ")".
            # Reshaping points to 3D as _trimBracketCharacters take data of shape (n_points, 1, n_times)
            points_new = self._trimBracketCharacters(points.reshape((points.shape[0], 1, 1)))
            self.points[points_dictkey] = points_new[..., 0]

    @timer
    def calculatePropertyMean(self, starttime=None, stoptime=None, get_tke_total=True, get_horizontal_mean=True, **kwargs):
        self.get_tke_total = get_tke_total
        self.get_horizontal_mean = get_horizontal_mean
        # Read points coordinates of each patch if also computing horizontal mean on top of temporal averaging
        if get_horizontal_mean: self._readPoints()
        # times_all in _selectTimes() should be sample_times in this case, thus temporarily change times_all to sample_times
        times_all_tmp = self.times_all.copy()
        self.times_all = self.sample_times
        # Find selected times and start, stop indices from sample_times
        self.times_selected, self.starttime_real, self.stoptime_real, istart, istop = self._selectTimes(starttime=starttime, stoptime=stoptime)
        # Switch times_all back
        self.times_all = times_all_tmp
        # Go through all properties and every patch
        for i in range(len(self.property_keys)):
            # Selected property data at selected times
            property_selected = self.data[self.property_keys[i]][:, :, istart:(istop + 1)]
            calc_tke_resolved = True if get_tke_total and 'U' in self.property_keys[i] else False
            # Keep in mind which patch current property lies in
            current_patch = self.property_keys[i].split('_')[1]
            # Property mean is sum(property_j*time_j)/sum(times)
            property_dot_dt_sum, dt_sum = 0., 0.
            tke_resolved_dot_dt_sum, dt_sum_tke_resolved = 0., 0.
            for j in range(1, len(self.times_selected)):
                # For subsequent times, dt = t_j - t_j-1
                dt = self.times_selected[j] - self.times_selected[j - 1]
                # Linear interpolation between each value point
                property_interp = (property_selected[:, :, j - 1] + property_selected[:, :, j])/2.
                property_dot_dt = property_interp*dt

                property_dot_dt_sum += property_dot_dt
                dt_sum += dt
                # In case current property is U and total mean TKE is asked,
                # calculate it from U' = U - <U>, TKE_resolved = 0.5U'U'
                # <TKE_resolved> = 0.5<U'U'>
                if calc_tke_resolved:
                    # <U> at current time correlation stage
                    u_mean_stage = property_dot_dt_sum/dt_sum
                    # U'U', instantaneous
                    uuprime2 = (property_interp - u_mean_stage)**2.
                    if self.debug: print("Instantaneous resolved TKE = {}".format(0.5*uuprime2))
                    # sum(TKE_resolved*dt)
                    tke_resolved_dot_dt_sum += 0.5*np.sum(uuprime2, axis=1, keepdims=True)*dt
                    # sum(dt)
                    dt_sum_tke_resolved = dt_sum

            # Store in dictionary
            self.data_mean[self.property_keys[i]] = property_dot_dt_sum/dt_sum
            if calc_tke_resolved: self.data_mean['kResolved_' + current_patch] = tke_resolved_dot_dt_sum/dt_sum_tke_resolved
            # If also perform horizontal averaging
            if get_horizontal_mean:
                # To get horizontal mean, sort z and get its sorted index
                idx_sorted = np.argsort(self.points[self.points_name + '_' + current_patch][:, 2])
                # Index to sort relevant arrays back to original order
                idx_revertsort = np.argsort(idx_sorted)
                # Sorted point coordinates of current patch
                points_sorted = self.points[self.points_name + '_' + current_patch][idx_sorted]
                # Then sort current property at current patch with this sorted index
                data_mean_sorted = self.data_mean[self.property_keys[i]][idx_sorted]
                # Do the same if resolved TKE is calculated
                if calc_tke_resolved: k_mean_sorted = self.data_mean['kResolved_' + current_patch][idx_sorted]
                n_points_per_lvl = len(points_sorted[points_sorted[:, 2] == points_sorted[0, 2]])
                # Go through every height level and do averaging
                ih = 0
                while ih < len(idx_sorted) - 1:
                    data_mean_sorted[ih:ih + n_points_per_lvl] = np.mean(data_mean_sorted[ih:ih + n_points_per_lvl], axis=0)
                    if calc_tke_resolved: k_mean_sorted[ih:ih + n_points_per_lvl] = np.mean(k_mean_sorted[ih:ih + n_points_per_lvl], axis=0)
                    ih += n_points_per_lvl

                # Sort sorted arrays back to original order
                self.data_mean[self.property_keys[i]] = data_mean_sorted[idx_revertsort]
                if calc_tke_resolved: self.data_mean['kResolved_' + current_patch] = k_mean_sorted[idx_revertsort]

        if self.debug and get_horizontal_mean:
            self.data_mean_sorted = data_mean_sorted
            self.points_sorted = points_sorted

        # If accumulated time of TKE resolved is not 0, add it to the existing key called "k", a.k.a. SGS TKE
        if get_tke_total:
            # data_mean_keys has the addition of "kResolved_{patch}" that will merge with "k_{patch}"
            self.data_mean_keys = list(self.data_mean.keys())
            # Go through properties to find mean resolved TKE of a patch
            for i in range(len(self.data_mean_keys)):
                if 'kResolved' in self.data_mean_keys[i]:
                    # Go through properties again to find mean SGS TKE of the same patch
                    for j in range(len(self.data_mean_keys)):
                        if 'k' in self.data_mean_keys[j] \
                            and 'kResolved' not in self.data_mean_keys[j] \
                            and self.data_mean_keys[j].split('_')[1] == self.data_mean_keys[i].split('_')[1]:
                            # data_mean_keys[j] is called k_{patch}
                            # data_mean_keys[i] is called kResolved_{patch}
                            self.data_mean[self.data_mean_keys[j]] += self.data_mean[self.data_mean_keys[i]]
                            print("Mean total TKE has been calculated for {}".format(self.data_mean_keys[j]))
                            break

    @timer
    def formatMeanDataToOpenFOAM(self, ke_relaxfactor=1.):
        # Go through inflow patches
        for i, patch in enumerate(self.inflow_patches):
            # Create time folders
            timefolder0 = self.avg_folder_patchpaths[i] + '0/'
            timefolder100000 = self.avg_folder_patchpaths[i] + '100000/'
            os.makedirs(timefolder0, exist_ok=True)
            os.makedirs(timefolder100000, exist_ok=True)
            # For each patch, go through (mean) properties
            for j in range(len(self.property_keys)):
                # Pick up only property corresponding current patch
                if patch in self.property_keys[j]:
                    # Get property name
                    property_name = self.property_keys[j].replace('_' + patch, '')
                    # Get mean property data
                    data_mean = self.data_mean[self.property_keys[j]]
                    if property_name in ('k', 'epsilon'): data_mean *= ke_relaxfactor
                    # Open file for writing
                    fid = open(timefolder0 + property_name, 'w')
                    print('Writing {0} to {1}'.format(property_name, timefolder0))
                    # Define datatype and average (placeholder) value
                    if property_name in ('k', 'T', 'pd', 'nuSGS', 'kappat', 'epsilon'):
                        datatype, average = 'scalar', '0'
                    else:
                        datatype, average = 'vector', '(0 0 0)'

                    # Write the file header
                    fid.write('/*--------------------------------*- C++ -*----------------------------------*\\\n')
                    fid.write('| =========                 |                                                 |\n')
                    fid.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                    fid.write('|  \\\\    /   O peration     | Version:  1.6                                   |\n')
                    fid.write('|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
                    fid.write('|    \\\\/     M anipulation  |                                                 |\n')
                    fid.write('\*---------------------------------------------------------------------------*/\n')
                    fid.write('FoamFile\n')
                    fid.write('{\n')
                    fid.write('    version     2.0;\n')
                    fid.write('    format      ascii;\n')
                    fid.write('    class       ')
                    fid.write(datatype)
                    fid.write('AverageField;\n')
                    fid.write('    object      values;\n')
                    fid.write('}\n')
                    fid.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n')
                    fid.write('// This inflow plane has been averaged from {0} s to {1} s\n'.format(self.starttime_real, self.stoptime_real))
                    if ke_relaxfactor < 1. and property_name in ('k', 'epsilon'): fid.write('// and relaxed to {} of full magnitude for time-varying inflow BC\n'.format(ke_relaxfactor))
                    if property_name == 'k' and self.get_tke_total: fid.write('// This is total TKE\n')
                    if self.get_horizontal_mean: fid.write('// Both temporal and horizontal averaging is done\n')
                    fid.write('// Min: {}, max: {}, mean: {}\n'.format(np.min(data_mean, axis=0), np.max(data_mean, axis=0), np.mean(data_mean, axis=0)))
                    fid.write('// Average\n')
                    fid.write(average)
                    fid.write('\n\n\n')
                    fid.write(str(data_mean.shape[0]))
                    fid.write('\n')
                    fid.write('(\n')
                    # Write property data
                    for k in range(data_mean.shape[0]):
                        # If U, replace comma with nothing
                        if property_name == 'U':
                            fid.write(str(tuple(data_mean[k])).replace(',', ''))
                        else:
                            fid.write(str(data_mean[k, 0]))

                        fid.write('\n')

                    fid.write(')')
                    fid.close()
                    # # Also copy files to 10000 time directory
                    # print('\nCopying {} to {}'.format(property_name, timefolder100000))
                    # shutil.copy(timefolder0 + property_name, timefolder100000)

        print("\nDon't forget to copy 'points' to {patch}/ folder too!")

if __name__ == '__main__':
    casename = 'ABL_N_H'
    casedir = '/media/yluan'
    boundarydata_folder = 'boundaryData_epsilon'  # 'boundaryData', 'boundaryData_epsilonTotal'
    filenames = "*"
    smooth_k = False
    n_timesample = 1000
    starttime, stoptime = 20000, 25000
    case = InflowBoundaryField(casename=casename, casedir=casedir, boundarydata_folder=boundarydata_folder, debug=True)
    case._readPoints()
    case.readPropertyData(filenames=filenames, n_timesample=n_timesample)
    case.calculatePropertyMean(starttime=starttime, stoptime=stoptime, get_tke_total=True)
    if smooth_k:
        idx_sorted_south = np.argsort(case.points['points_south'][:, 2])
        idx_revertsort_south = np.argsort(idx_sorted_south)
        points_sorted_south = case.points['points_south'][idx_sorted_south]
        idx_sorted_west = np.argsort(case.points['points_west'][:, 2])
        idx_revertsort_west = np.argsort(idx_sorted_west)
        points_sorted_west = case.points['points_west'][idx_sorted_west]
        k_south_sorted = case.data_mean['k_south'][idx_sorted_south]
        k_west_sorted = case.data_mean['k_west'][idx_sorted_west]

        kmin_south_i = np.where(k_south_sorted == min(k_south_sorted))[0][0]
        z_kmin_south = points_sorted_south[:, 2][kmin_south_i]
        kmin_west_i = np.where(k_west_sorted == min(k_west_sorted))[0][0]
        z_kmin_west = points_sorted_west[:, 2][kmin_west_i]
        zmax_south = zmax_west = 1000.

        z_uniq_south = np.unique(points_sorted_south[:, 2])
        z_uniq_west = np.unique(points_sorted_west[:, 2])
        z_uniq_south = z_uniq_south[z_uniq_south >= z_kmin_south]
        z_uniq_west = z_uniq_west[z_uniq_west >= z_kmin_west - 10.]

        kdiff_south = k_south_sorted[-1] - k_south_sorted[kmin_south_i]
        kdiff_west = k_west_sorted[-1] - k_west_sorted[kmin_west_i - 300]
        dk_south = kdiff_south/len(z_uniq_south)
        dk_west = kdiff_west/len(z_uniq_west)
        for i in range(kmin_south_i + 300, len(k_south_sorted) - 300):
            # if k_south_sorted[i] > k_south_sorted[-1]: k_south_sorted[i] = k_south_sorted[-1]
            k_south_sorted[i:i + 300] = k_south_sorted[i - 300] + dk_south

        for i in range(kmin_west_i, len(k_west_sorted) - 300):
            # if k_west_sorted[i] > k_west_sorted[-1]: k_west_sorted[i] = k_west_sorted[-1]
            k_west_sorted[i:i + 300] = k_west_sorted[i - 300] + dk_west

        case.data_mean['k_south'] = k_south_sorted[idx_revertsort_south]
        case.data_mean['k_west'] = k_west_sorted[idx_revertsort_west]

    case.formatMeanDataToOpenFOAM(ke_relaxfactor=1)

    from PlottingTool import Plot2D
    idx_sorted = np.argsort(case.points['points_south'][:, 2])
    points_sorted = case.points['points_south'][:, 2][idx_sorted]/750.
    idx_sorted2 = np.argsort(case.points['points_west'][:, 2])
    points_sorted2 = case.points['points_west'][:, 2][idx_sorted2]/750.
    for name in case.filenames:
        if name == 'k':
            xlabel = r'$\langle k \rangle$ [m$^2$/s$^2$]'
        elif name == 'U':
            xlabel = r'$\langle U \rangle$ [m/s]'
        elif name == 'T':
            xlabel = r'$\langle T \rangle$ [K]'
        elif name == 'epsilon':
            xlabel = r'$\langle \epsilon_{\mathrm{SGS}} \rangle$ [m$^2$/s$^3$]'

        data_sorted = case.data_mean[name + '_south'][idx_sorted]
        data_sorted2 = case.data_mean[name + '_west'][idx_sorted2]
        xlim = (min(min(data_sorted.ravel()), min(data_sorted2.ravel())) - 0.1*min(data_sorted.ravel()),
                max(max(data_sorted.ravel()), max(data_sorted2.ravel())) + 0.1*max(data_sorted.ravel()))
        # Use magnitude
        if name == 'U':
            data_sorted = np.sqrt(data_sorted[:, 0]**2 + data_sorted[:, 1]**2 + data_sorted[:, 2]**2)
            data_sorted2 = np.sqrt(data_sorted2[:, 0]**2 + data_sorted2[:, 1]**2 + data_sorted2[:, 2]**2)

        listx = (data_sorted, data_sorted2)
        listy = (points_sorted, points_sorted2)

        myplot = Plot2D(listx, listy, plot_type='infer',
                        show=False, save=True, name=name, xlabel=xlabel, ylabel=r'$\frac{z}{z_i}$ [-]',
                        figdir=case.avg_folder_path, figwidth='1/3', xlim=xlim)
        myplot.initializeFigure()
        myplot.plotFigure(linelabel=('South', 'West'))
        myplot.axes.fill_between(xlim, 27/750., 153/750., alpha=0.25)
        myplot.finalizeFigure()

    kappa = .4
    uref = 8.
    zref = 90.
    z0 = .2
    cmu = 0.03
    ustar = kappa*uref/np.log((zref + z0)/z0)
    u = ustar/kappa*np.log((case.points['points_south'][:, 2][idx_sorted] + z0)/z0)
    k = ustar**2/np.sqrt(cmu)
    epsilon = ustar**3/kappa/(case.points['points_south'][:, 2][idx_sorted] + z0)

    ulim = (min(u) - 0.1*min(u),
            max(u) + 0.1*max(u))
    epslim = (min(epsilon) - 0.1*min(epsilon),
            max(epsilon) + 0.1*max(epsilon))
    myplot = Plot2D(u, points_sorted, plot_type='infer',
                    show=False, save=True, name='U_atmBC', xlabel='U [m/s]', ylabel="z/zi [-]",  # r'$\frac{z}{z_i}$ [-]',
                    figdir=case.avg_folder_path, figwidth='1/3', xlim=ulim)
    myplot.initializeFigure()
    myplot.plotFigure()  # linelabel=('South', 'West'))
    myplot.axes.fill_between(xlim, 27/750., 153/750., alpha=0.25)
    myplot.finalizeFigure()

    myplot = Plot2D(epsilon, points_sorted, plot_type='infer',
                    show=False, save=True, name='epsilon_atmBC', xlabel='\epsilon [m2/s3]', ylabel=r'$\frac{z}{z_i}$ [-]',
                    figdir=case.avg_folder_path, figwidth='1/3', xlim=epslim)
    myplot.initializeFigure()
    myplot.plotFigure()
    myplot.axes.fill_between(xlim, 27/750., 153/750., alpha=0.25)
    myplot.finalizeFigure()



    # kSouth = case.data['k_south']
    # uSouth = case.data['U_south']






    # casename = 'ALM_N_H_ParTurb'
    # filenames = 'Cd'
    # startTime1 = 20000
    # stopTime1 = 22000
    # frameSkip = 182#28
    #
    # turb = TurbineOutputs(casename = casename, casedir = '/media/yluan/Toshiba External Drive')
    #
    # turb.readPropertyData(filenames = filenames)
    #
    # turb.calculatePropertyMean(starttime = startTime1, stoptime = stopTime1)
    #
    # listX1 = (turb.times_selected[::frameSkip],)*3
    # listY1 = (turb.data[filenames + '_Turb0_Bld0_mean'][::frameSkip],
    #          turb.data[filenames + '_Turb0_Bld1_mean'][::frameSkip],
    #          turb.data[filenames + '_Turb0_Bld2_mean'][::frameSkip])
    # listY2 = (turb.data[filenames + '_Turb1_Bld0_mean'][::frameSkip],
    #           turb.data[filenames + '_Turb1_Bld1_mean'][::frameSkip],
    #           turb.data[filenames + '_Turb1_Bld2_mean'][::frameSkip])
    #
    # startTime2 = 21000
    # stopTime2 = 22000
    # turb.calculatePropertyMean(starttime = startTime2, stoptime = stopTime2)
    #
    # listX2 = (turb.times_selected[::frameSkip],)*3
    # listY3 = (turb.data[filenames + '_Turb0_Bld0_mean'][::frameSkip],
    #           turb.data[filenames + '_Turb0_Bld1_mean'][::frameSkip],
    #           turb.data[filenames + '_Turb0_Bld2_mean'][::frameSkip])
    # listY4 = (turb.data[filenames + '_Turb1_Bld0_mean'][::frameSkip],
    #           turb.data[filenames + '_Turb1_Bld1_mean'][::frameSkip],
    #           turb.data[filenames + '_Turb1_Bld2_mean'][::frameSkip])
    #
    # figDir = '/media/yluan/Toshiba External Drive/' + casename + '/turbineOutput/Result'
    #
    # # Custom colors
    # colors, _ = Plot2D.setColors()
    #
    # plotsLabel = ('Blade 1', 'Blade 2', 'Blade 3')
    # transparentBg = False
    # xLim1 = (startTime1, stopTime1)
    # yLim = (min(np.min(listY1), np.min(listY2), np.min(listY3), np.min(listY4)), max(np.max(listY1), np.max(listY2), np.max(listY3), np.max(listY4)))
    #
    # show = False
    #
    # clPlot = Plot2D(listY1, listX1, save = True, name = 'Turb0_' + filenames  + '1', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]', figDir = figDir, xLim = yLim, yLim = xLim1, figWidth = 'half', figHeightMultiplier = 2., show = show, colors = colors[:3][:], gradientBg = True, gradientBgRange = (startTime1, 21800), gradientBgDir = 'y')
    # clPlot.initializeFigure()
    #
    # clPlot.plotFigure(plotsLabel = plotsLabel)
    #
    # clPlot.finalizeFigure(transparentBg = transparentBg)
    #
    # # clPlot2 = Plot2D(listX1, listY2, save = True, name = 'Turb1_' + filenames  + '1', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]', figDir = figDir, xLim = xLim1, yLim = yLim, figWidth = 'full', show = show, colors = colors[3:6][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # # clPlot2.initializeFigure()
    # # clPlot2.plotFigure(plotsLabel = plotsLabel)
    # # clPlot2.finalizeFigure(transparentBg = transparentBg)
    # #
    # #
    # #
    # #
    # #
    # #
    # # xLim2 = (startTime2, stopTime2)
    # #
    # # show = True
    # #
    # # clPlot = Plot2D(listX2, listY3, save = True, name = 'Turb0_' + filenames + '2', xLabel = 'Time [s]', yLabel = r'$C_d$ [-]',
    # #                 figDir = figDir, xLim = xLim2, yLim = yLim, figWidth = 'full', show = show, colors = colors[:3][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # # clPlot.initializeFigure()
    # #
    # # clPlot.plotFigure(plotsLabel = plotsLabel)
    # #
    # # clPlot.finalizeFigure(transparentBg = transparentBg)
    # #
    # # clPlot2 = Plot2D(listX2, listY4, save = True, name = 'Turb1_' + filenames + '2', xLabel = 'Time [s]',
    # #                  yLabel = r'$C_d$ [-]', figDir = figDir, xLim = xLim2, yLim = yLim, figWidth = 'full', show = show, colors = colors[3:6][:], gradientBg = True, gradientBgRange = (startTime1, 21800))
    # # clPlot2.initializeFigure()
    # # clPlot2.plotFigure(plotsLabel = plotsLabel)
    # # clPlot2.finalizeFigure(transparentBg = transparentBg)
