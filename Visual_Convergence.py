import numpy as np
from PrecursorAndTurbineOutputs import BaseProperties
from PlottingTool import Plot2D
from numba import prange

"""
User Inputs
"""
casedir = '/media/yluan'
casename = 'ALM_N_L_SeqTurb'
filenames = 'uuPrime2'  # 'uuPrime2', 'UAvg', 'Rmean'
# Which column is time
timeCol = 0  # 'infer'
# Characters to remove before reading data into numpy arrays
invalid_chars = ('(', ')')
times = (18500, 23000) if ('_L_' in casename or 'HiSpeed' in casename) else (20500, 25000)
# [CAUTION] Whether remerge all time directories into 1 ensemble even if a current ensemble exists
force_remerge = True


"""
Plot Settings
"""
saveFig = True
xlabel = 'Time [s]'
# ylabel = r'$R_{11}$'
xlim = times
figwidth = '1/3'
show = False
grad_bg, grad_bgRange = True, (times[0], times[1])


"""
Process User Inputs
"""
if casename == 'ALM_N_H_OneTurb':
    nProbe = 8
elif 'ParTurb' in casename:
    nProbe = 16
else:
    nProbe = 14

if 'ParTurb' in casename:
    linelabels = ('Turb0 + $1D$', 'Turb1 + $1D$',
                  'Turb0 + $4D$', 'Turb1 + $4D$')
elif casename == 'ALM_N_H_OneTurb':
    linelabels = ('$+1D$', '$+2D$', '$+4D$')
elif 'SeqTurb' in casename:
    linelabels = ('Turb0 + $1D$', 'Turb0 + $4D$',
                  'Turb1 + $1D$', 'Turb1 + $4D$')


"""
Merge Time Directories and Read Properties
"""
probe = BaseProperties(casename=casename + '/Probes', casedir=casedir, timecols=timeCol, force_remerge=force_remerge)

# From now on all files are read through Ensemble regardless
# Trim invalid '(' and ')' characters
probe.trimInvalidCharacters(filenames=filenames, invalid_chars=invalid_chars)

# Read properties from Ensemble
probe.readPropertyData(filenames=filenames, skipcol=timeCol + 1)

# # Calculate mean
# probe.calculatePropertyMean()


"""
Regroup Data Into xx, yy, zz or x, y, z Categories and Plot
"""
# Function to decompose data into xx, yy, zz or x, y, z components and plot
def decomposeDataAndPlot(step, subscript, startcol, filename):
    # For xx component
    property_decomposed = {}
    list_y = []
    for i in range(nProbe):
        property_decomposed['probe' + str(i)] = probe.data[filename][:, startcol + step*i]
        list_y.append(property_decomposed['probe' + str(i)])

    if casename == 'ALM_N_H_OneTurb':
        list_y = [list_y[3], list_y[5], list_y[7]]
    elif 'ParTurb' in casename:
        list_y = [list_y[5], list_y[7], list_y[13], list_y[15]]
    # For SeqTurb
    else:
        list_y = [list_y[3], list_y[7], list_y[9], list_y[13]]

    nplot = 1
    nprobe_plot = (0, 3) if casename == 'ALM_N_H_OneTurb' else (0, 4)
    for i in range(nplot):
        ylim = (np.min(list_y) - np.abs(np.min(list_y)*0.05),
                np.max(list_y) + np.abs(np.max(list_y)*0.05))
        plot = Plot2D(list_x[nprobe_plot[i]:nprobe_plot[i + 1]], list_y[nprobe_plot[i]:nprobe_plot[i + 1]], save=saveFig, name=filename + '_' + subscript + '_Convergence_' + str(i), xlabel=xlabel,
                      ylabel=ylabel,
                      figdir=figdir, xlim=xlim, ylim=ylim, figwidth=figwidth, show=show, grad_bg=grad_bg)

        plot.initializeFigure()
        plot.plotFigure(linelabel=linelabels[nprobe_plot[i]:nprobe_plot[i + 1]])
        plot.finalizeFigure()

def decomposeHorizontalAndPlot(step, subscript, startcol, filename):
    # For xx component
    property_decomposed = {}
    list_y = []
    for i in range(nProbe):
        property_decomposed['probe' + str(i)] = np.sqrt(probe.data[filename][:, startcol + step*i]**2. + probe.data[filename][:, startcol + 1 + step*i]**2.)
        list_y.append(property_decomposed['probe' + str(i)])

    # list_y = np.array(list_y)
    if casename == 'ALM_N_H_OneTurb':
        # 3: +1D turb apex, 5: +2D turb apex, 7: +4D turb apex
        list_y = [list_y[3], list_y[5], list_y[7]]
    elif 'ParTurb' in casename:
        # 5: southern +1D, 7:northern +1D, 13: southern +4D, 15: northern +4D
        list_y = [list_y[5], list_y[7], list_y[13], list_y[15]]
    # For SeqTurb
    else:
        # 3: upwind +1D, 7: upwind +4D, 9: downwind +1D, 13: downwind + 4D
        list_y = [list_y[3], list_y[7], list_y[9], list_y[13]]

    nplot = 1
    nprobe_plot = (0, 3) if casename == 'ALM_N_H_OneTurb' else (0, 4)
    for i in range(nplot):
        ylim = (np.min(list_y) - np.abs(np.min(list_y)*0.05),
                np.max(list_y) + np.abs(np.max(list_y)*0.05))
        plot = Plot2D(list_x[nprobe_plot[i]:nprobe_plot[i + 1]], list_y[nprobe_plot[i]:nprobe_plot[i + 1]], save=saveFig, name=filename + '_' + subscript + '_Convergence_' + str(i), xlabel=xlabel,
                      ylabel=ylabel,
                      figdir=figdir, xlim=xlim, ylim=ylim, figwidth=figwidth, show=show, grad_bg=grad_bg)
        plot.initializeFigure()
        plot.plotFigure(linelabel=linelabels[nprobe_plot[i]:nprobe_plot[i + 1]])
        plot.finalizeFigure()

def computeEigenValueAndPlot(step, subscript, startcol, filename, which_eigval=0):
    list_y = []
    for i in range(nProbe):
        tensor = probe.data[filename][:, startcol + step*i:startcol + 6 + step*i]
        tensor_full = np.empty((len(tensor), 9))
        tensor_full[:, :3] = tensor[:, :3]
        tensor_full[:, 3] = tensor[:, 1]
        tensor_full[:, 4:6] = tensor[:, 3:5]
        tensor_full[:, 6] = tensor[:, 2]
        tensor_full[:, 7] = tensor[:, 4]
        tensor_full[:, 8] = tensor[:, 5]
        tensor_full = tensor_full.reshape((-1, 3, 3))
        eigval = np.empty((len(tensor), 3))
        for j in range(len(tensor)):
            eigval[j], _ = np.linalg.eigh(tensor_full[j])

        list_y.append(eigval[:, which_eigval])

    if casename == 'ALM_N_H_OneTurb':
        list_y = [list_y[3], list_y[5], list_y[7]]
    elif 'ParTurb' in casename:
        list_y = [list_y[5], list_y[7], list_y[13], list_y[15]]
    # For SeqTurb
    else:
        list_y = [list_y[3], list_y[7], list_y[9], list_y[13]]

    nplot = 1
    nprobe_plot = (0, 3) if casename == 'ALM_N_H_OneTurb' else (0, 4)
    for i in prange(nplot):
        ylim = (np.min(list_y) - np.abs(np.min(list_y)*0.05),
                np.max(list_y) + np.abs(np.max(list_y)*0.05))
        plot = Plot2D(list_x[nprobe_plot[i]:nprobe_plot[i + 1]], list_y[nprobe_plot[i]:nprobe_plot[i + 1]], save=saveFig, name='lambda_' + subscript + '_Convergence_' + str(i), xlabel=xlabel,
                      ylabel=ylabel,
                      figdir=figdir, xlim=xlim, ylim=ylim, figwidth=figwidth, show=show, grad_bg=grad_bg)
        plot.initializeFigure()
        plot.plotFigure(linelabel=linelabels[nprobe_plot[i]:nprobe_plot[i + 1]])
        plot.finalizeFigure()

figdir = probe.case_fullpath + 'Result'
# list_x = [probe.timesSelected]*nProbe
list_x = (probe.times_all,)*nProbe
if 'uuPrime2' in filenames:
    """
    Reynolds Stress Plots
    """
    filename = 'uuPrime2'
    # Pick up columns every 6 steps due to symmetric tensor saved in the order of
    # (xx, xy, xz, yy, yz, zz)
    step, startcol = 6, 0

    ylabel = r'$\langle \lambda_{1} \rangle$ [m$^2$/s$^2$]'
    computeEigenValueAndPlot(step, '1', startcol, filename, 0)

    ylabel = r'$\langle \lambda_{2} \rangle$ [m$^2$/s$^2$]'
    computeEigenValueAndPlot(step, '2', startcol, filename, 1)

    ylabel = r'$\langle \lambda_{3} \rangle$ [m$^2$/s$^2$]'
    computeEigenValueAndPlot(step, '3', startcol, filename, 2)

    # # Decompose and plot xx component
    # ylabel = r'$\langle \tau_{1} \rangle$ [m$^2$/s$^2$]'
    # decomposeDataAndPlot(step=step, subscript='11', startcol=startcol, filename=filename)
    #
    # # Decompose and plot xy component
    # ylabel = r'$\langle \tau_{12} \rangle$ [m$^2$/s$^2$]'
    # startcol = 1
    # decomposeDataAndPlot(step=step, subscript='12', startcol=startcol, filename=filename)
    #
    # # Decompose and plot xz component
    # ylabel = r'$\langle \tau_{13} \rangle$ [m$^2$/s$^2$]'
    # startcol = 2
    # decomposeDataAndPlot(step=step, subscript='13', startcol=startcol, filename=filename)

    # # Decompose and plot yy component
    # startcol = 3
    # ylabel = r'$\langle \tau_{22} \rangle$ [m$^2$/s$^2$]'
    # decomposeDataAndPlot(step=step, subscript='22', startcol=startcol, filename=filename)
    #
    # # Decompose and plot yz component
    # ylabel = r'$\langle \tau_{23} \rangle$ [m$^2$/s$^2$]'
    # startcol = 4
    # decomposeDataAndPlot(step=step, subscript='23', startcol=startcol, filename=filename)
    #
    # # Decompose and plot yy component
    # startcol = 5
    # ylabel = r'$\langle \tau_{33} \rangle$ [m$^2$/s$^2$]'
    # decomposeDataAndPlot(step=step, subscript='33', startcol=startcol, filename=filename)


if 'UAvg' in filenames:
    """
    UAvg Plots
    """
    filename = 'UAvg'
    step, startcol = 3, 0

    # Decompose and plot x component
    ylabel = r'$\langle \tilde{U}_\mathrm{hor} \rangle$ [m/s]'
    # decomposeDataAndPlot(step = step, subscript = '1', startcol = startcol, filename = filename)
    decomposeHorizontalAndPlot(step=step, subscript='hor', startcol=startcol, filename=filename)

    # # Decompose and plot y component
    # startcol = 1
    # ylabel = r'$\langle U_{y} \rangle$ [m/s]'
    # decomposeDataAndPlot(step = step, subscript = '2', startcol = startcol, filename = filename)

    # Decompose and plot z component
    startcol = 2
    ylabel = r'$\langle \tilde{u}_z \rangle$ [m/s]'
    decomposeDataAndPlot(step = step, subscript = 'z', startcol = startcol, filename = filename)


if 'Rmean' in filenames:
    """
    SFS Deviatoric Stress Plots
    """
    filename = 'Rmean'
    # Pick up columns every 6 steps due to symmetric tensor saved in the order of
    # (xx, xy, xz, yy, yz, zz)
    step, startcol = 6, 0

    # Decompose and plot xx component
    ylabel = r'$R_{11}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '11', startcol = startcol, filename = filename)

    # Decompose and plot yy component
    startcol = 3
    ylabel = r'$R_{22}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '22', startcol = startcol, filename = filename)

    # Decompose and plot yy component
    startcol = 5
    ylabel = r'$R_{33}$ [m$^2$/s$^2$]'
    decomposeDataAndPlot(step = step, subscript = '33', startcol = startcol, filename = filename)










