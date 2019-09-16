import os
import numpy as np
from warnings import warn
from Utility import timer
import shutil

class BaseProperties:
    def __init__(self, casename, casedir='.', filename_pre='', filename_sub='', ensemblefolder_name='Ensemble', result_folder='Result', timecols='infer', time_kw='ime', force_remerge=False, **kwargs):
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
        times_selected = self.times_all[istart:istop]

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
    def __init__(self, casename, height_filename='hLevelsCell', bl_folder='ABL', **kwargs):
        self.height_filename = height_filename
        super(BoundaryLayerProfiles, self).__init__(casename = casename + '/' + bl_folder, timecols = 0, excl_files =
        height_filename, **kwargs)
        # Copy height_filename to Ensemble in order to use it later
        time = os.listdir(self.case_fullpath)[0]
        shutil.copy2(self.case_fullpath + time + '/' + height_filename, self.ensemble_folderpath)


    def readPropertyData(self, filenames = ('*'), **kwargs):
        # Read height levels
        self.hLvls = np.genfromtxt(self.ensemble_folderpath + self.height_filename)
        # Override skipcol to suit inflow property files
        # Columns to skip are 0: time; 1: time step
        super(BoundaryLayerProfiles, self).readPropertyData(filenames = filenames, skipcol = 2)


    def calculatePropertyMean(self, starttime = None, stoptime = None, **kwargs):
        # Override axis to suit inflow property files
        super(BoundaryLayerProfiles, self).calculatePropertyMean(axis = 0, starttime = starttime, stoptime = stoptime, **kwargs)


class TurbineOutputs(BaseProperties):
    def __init__(self, casename, dataFolder = 'turbineOutput', globalQuantities = ('powerRotor', 'rotSpeed', 'thrust',
                                                                      'torqueRotor',
                                                       'torqueGen', 'azimuth', 'nacYaw', 'pitch'), **kwargs):
        self.globalQuantities = globalQuantities
        super(TurbineOutputs, self).__init__(casename + '/' + dataFolder, **kwargs)

        self.nTurb, self.nBlade = 0, 0


    @timer
    def readPropertyData(self, filenames = ('*',), skiprow = 0, skipcol = 'infer', verbose = True, turbInfo = ('infer',)):
        filenames = self._ensureTupleInput(filenames)
        globalQuantities = (
        'powerRotor', 'rotSpeed', 'thrust', 'torqueRotor', 'torqueGen', 'azimuth', 'nacYaw', 'pitch', 'powerGenerator')
        if skipcol is 'infer':
            skipcol = []
            for file in filenames:
                if file in globalQuantities:
                    skipcol.append(3)
                else:
                    skipcol.append(4)

        super(TurbineOutputs, self).readPropertyData(filenames = filenames, skiprow = skiprow, skipcol = skipcol)

        if turbInfo[0] is 'infer':
            turbInfo = np.genfromtxt(self.ensemble_folderpath + self.filename_pre + 'Cl' + self.filename_sub)[skiprow:, :2]

        # Number of turbines and blades
        (self.nTurb, self.nBlade) = (int(np.max(turbInfo[:, 0]) + 1), int(np.max(turbInfo[:, 1]) + 1))

        fileNamesOld, self.filenames = self.filenames, list(self.filenames)
        for filename in fileNamesOld:
            for i in range(self.nTurb):
                if filename not in globalQuantities:
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
    def __init__(self, casename, casedir='.', boundarydata_folder='boundaryData', avg_folder='Average', **kwargs):
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

    @timer
    def readPropertyData(self, filenames=('*',), skiprow=22, skipfooter=1, n_timesample=-1, lstr_precision=12, rstr_precision=20):
        def __trimBracketCharacters(data):
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
                    data_new = __trimBracketCharacters(data)
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

    @timer
    def calculatePropertyMean(self, starttime=None, stoptime=None, **kwargs):
        # times_all in _selectTimes() should be sample_times in this case, thus temporarily change times_all to sample_times
        times_all_tmp = self.times_all.copy()
        self.times_all = self.sample_times
        # Find selected times and start, stop indices from sample_times
        self.times_selected, self.starttime_real, self.stoptime_real, istart, istop = self._selectTimes(starttime=starttime, stoptime=stoptime)
        # Switch times_all back
        self.times_all = times_all_tmp
        # Go through all properties
        for i in range(len(self.property_keys)):
            # Selected property data at selected times
            propert_selected = self.data[self.property_keys[i]][:, :, istart:(istop + 1)]
            # Property mean is sum(property_j*time_j)/sum(times)
            property_dot_time_sum = 0.
            for j in range(len(self.times_selected)):
                property_dot_time = np.multiply(propert_selected[:, :, j], self.times_selected[j])
                property_dot_time_sum += property_dot_time

            # Store in dictionary
            self.data_mean[self.property_keys[i]] = property_dot_time_sum/np.sum(self.times_selected)

    @timer
    def formatMeanDataToOpenFOAM(self):
        # Go through inflow patches
        for i, patch in enumerate(self.inflow_patches):
            # Create time folders
            timefolder0 = self.avg_folder_patchpaths[i] + '0/'
            timefolder10000 = self.avg_folder_patchpaths[i] + '10000/'
            os.makedirs(timefolder0, exist_ok=True)
            os.makedirs(timefolder10000, exist_ok=True)
            # For each patch, go through (mean) properties
            for j in range(len(self.property_keys)):
                # Pick up only property corresponding current patch
                if patch in self.property_keys[j]:
                    # Get property name
                    property_name = self.property_keys[j].replace('_' + patch, '')
                    # Get mean property data
                    data_mean = self.data_mean[self.property_keys[j]]
                    # Open file for writing
                    fid = open(timefolder0 + property_name, 'w')
                    print('Writing {0} to {1}'.format(property_name, timefolder0))
                    # Define datatype and average (placeholder) value
                    if property_name in ('k', 'T', 'pd', 'nuSGS', 'kappat'):
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
                    # Also copy files to 10000 time directory
                    print('\nCopying {} to {}'.format(property_name, timefolder10000))
                    shutil.copy(timefolder0 + property_name, timefolder10000)

        print("\nDon't forget to copy 'points' to {patch}/ folder too!")

if __name__ == '__main__':
    from PlottingTool import Plot2D

    casename = 'ABL_N_H'
    casedir = '/media/yluan'
    filenames = '*'
    n_timesample = 1000
    starttime, stoptime = 20000, 25000
    case = InflowBoundaryField(casename=casename, casedir=casedir)
    case.readPropertyData(filenames=filenames, n_timesample=n_timesample)
    case.calculatePropertyMean(starttime=starttime, stoptime=stoptime)
    case.formatMeanDataToOpenFOAM()

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
