import numpy as np
import os

class SetProperties(object):
    def __init__(self, casename='ALM_N_H_OneTurb', casedir='/media/yluan',
                 time='latestTime', resultfolder='Result', setfolder='Sets'):
        self.case_path = casedir + '/' + casename + '/'
        self.set_path = self.case_path + setfolder + '/'
        self.result_path = self.set_path + resultfolder + '/'
        os.makedirs(self.result_path, exist_ok=True)
        # If time is 'latestTime', take the last time while excl. result folder
        self.time = os.listdir(self.set_path)[-2] if 'latest' in time else str(time)
        self.time_path = self.set_path + self.time + '/'
        self.result_path += self.time + '/'
        os.makedirs(self.result_path, exist_ok=True)

    def readSets(self, orientation_kw='_H_', ext='.xy'):
        self.sets = os.listdir(self.time_path)
        self.orientation = {}
        self.coor, self.data = {}, {}
        for i, set in enumerate(self.sets):
            val = np.genfromtxt(self.time_path + set)
            # Remove extension string
            set = set.replace(ext, '')
            self.sets[i] = set
            self.orientation[set] = 'H' if orientation_kw in set else 'V'
            self.coor[set] = val[:, 0]
            self.data[set] = val[:, 1:]



