from PrecursorAndTurbineOutputs import TurbineOutputs
from PlottingTool import Plot2D
import numpy as np

"""
Calculate Power Ratio Between Downwind and Upwind Turbines
"""
# casedir = '/media/yluan'
# casename = 'ALM_N_H_OneTurb'
# property_names = 'powerGenerator'
# times = (18500, 23000) if ('_L_' in casename or 'HiSpeed' in casename) else (20500, 25000)
# # [CAUTION] Whether remerge all time directories into 1 ensemble even if a current ensemble exists
# force_remerge = True
#
# # Initialize turb object
# turb = TurbineOutputs(casename=casename, casedir=casedir, force_remerge=force_remerge)
#
# # Read property data
# turb.readPropertyData(filenames = property_names)
#
# # Calculate time averaged mean between times, thus axis is row
# turb.calculatePropertyMean(starttime = times[0], stoptime = times[1], axis = 'row')
#
# power_ratio = turb.data[property_names + '_Turb1_mean']/turb.data[property_names + '_Turb0_mean']
#
# print('\nPower ratio between downwind front turbine and upwind turbine is {}'.format(power_ratio))


"""
User Inputs
"""
casedir = '/media/yluan/RANS'
casenames = ('N_L_SeqTurb_600m',)  # 'N_H_ParTurb_LowZ_Rwall', 'N_H_OneTurb_LowZ_Rwal2'
property_names = ('powerGenerator', 'thrust')
# [CAUTION] Whether remerge all time directories into 1 ensemble even if a current ensemble exists
force_remerge = False
times = (0, 11000)
# times = (0, 12000)
xlabel = {property_names[0]: 'Power [KW]',
          property_names[1]: 'Thrust [KN]'}
# Turbines are 9.1552 rpm => 6.554 s/r. Time step is 0.036 s/step => 182.046 steps/r
frameskip = 1  # 182  # steps/r
# First 4 columns are not property data; next 6 columns are 0 for Cl and Cd
property_colskip = 3  # Default 4

"""
Plot Settings
"""
# Figure width is half of A4 page, height is multiplied to elongate it
figwidth, figheight_multiplier = 'half', 1.
show, save = False, True

ylabel = 'Iteration [-]'
grad_bg = True
xlim = ((0, 2500), (0, 400)) if 'HiSpeed' not in casenames[0] else ((0, 4700), (0, 600))
target_thrust = 365.96 if 'ParTurb' in casenames[0] else 371.4
target_pwr = 2222.64 if 'ParTurb' in casenames[0] else 2275.85
if 'SeqTurb' in casenames[0]: target_pwr, target_thrust = 1550., 295.


"""
Read Property Data and Plot
"""
# Go through cases
for casename in casenames:
    target = (target_pwr, target_thrust)
    ylim = (times[0], times[1])
    if grad_bg:
        grad_bg_dir, grad_bg_range = 'x', (times[0], times[0] + 0.8*(times[1] - times[0]))
    else:
        grad_bg_dir, grad_bg_range = 'x', None

    linelabel = ("RANS", 'Target') if 'OneTurb' in casename else ('Average', 'Turb0', 'Turb1', 'Target')
    figdir = casedir + '/' + casename + '/turbineOutput/Result'
    # Initialize turb object
    turb = TurbineOutputs(casename=casename, casedir=casedir, force_remerge=force_remerge, timecols=2)

    # Read property data
    turb.readPropertyData(filenames=property_names, skipcol=property_colskip)

    # # Calculate ALM segment averaged mean between times
    # turb.calculatePropertyMean(starttime=times[0], stoptime=times[1])

    # Actual times in a list of length of number of lines in a figure
    list_y = (turb.times_all,)*(turb.n_turbs + 2) if 'OneTurb' not in casename else (turb.times_all,)*2
    # Go through properties
    for i0, property in enumerate(property_names):
        # Properties dictionary in a list of length of number of lines in a figure
        list_x = []
        # Initialize extreme xlim that gets updated in the following turbine loop
        # xlim = (1e9, -1e9)
        # Go through turbines
        total = np.zeros(len(turb.times_all))
        for i in range(turb.n_turbs):
            mean_property = property + '_Turb' + str(i)
            list_x.append(turb.data[mean_property].ravel()/1000.)
            # xlim gets updated every turbine so all turbines are considered in the end
            # xlim = (min(xlim[0], np.min(list_x[i])), max(xlim[1], np.max(list_x[i])))
            total += turb.data[mean_property].ravel()/1000.

        if 'OneTurb' not in casename:
            list_x.append(total/turb.n_turbs)
            list_x = [list_x[-1], list_x[0], list_x[1]]

        list_x.append(np.ones(len(total))*target[i0])

        """
        Plot for every turbine for this property
        """
        # Initialize figure object
        # In this custom color, blades of different turbines are of different colors
        property_plot = Plot2D(list_y, list_x, save=save, name=property, xlabel=ylabel, ylabel=xlabel[property], figdir=figdir, xlim=ylim, ylim=xlim[i0], figwidth=figwidth, figheight_multiplier=figheight_multiplier, show=show, grad_bg=grad_bg, grad_bg_range=grad_bg_range, grad_bg_dir=grad_bg_dir)

        # Create the figure window
        property_plot.initializeFigure()

        # Plot the figure
        property_plot.plotFigure(linelabel=linelabel)

        # Finalize figure
        property_plot.finalizeFigure()








