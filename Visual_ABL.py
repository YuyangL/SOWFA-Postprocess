import matplotlib.pyplot as plt
import numpy as np
# from warnings import warn
from PrecursorAndTurbineOutputs import BoundaryLayerProfiles
from PlottingTool import Plot2D
from Utilities import readData
from numba import prange

"""
User Inputs
"""
# casedir, casename = 'J:', 'Doekemeijer/neutral_3kmx3kmx1km_lowTI_225deg_11mps'
casedir, casename = '/media/yluan', 'ABL_N_H'
# Profile of interest
profile = 'U'  # 'U', 'T', 'heatFlux', 'TI'
starttime, stoptime = 18000, 22000
# Hub height velocity, hub height, rotor D
Uhub, zHub, D = 8., 90., 126.
# Inversion layer height, inversion layer width
zi, invW = 750., 100.
# Whether to calculate zi from q3_mean and Tw_mean, will override existing zi
calc_zi = False
# Force to remerge all time directories to Ensemble folder?
force_remerge=False  # [CAUTION]
refDataFolder, refDataFormat = 'Churchfield', '.csv'
# Unused
startTime2, stopTime2 = 3000, 6000


"""
Plot Settings
"""
show, save = False, True
# Fill in area alpha for rotor swept area and capping inversion layer
fillAlpha = 0.25


"""
Process User Inputs
"""
if profile == 'U':
    filenames = ['U_mean', 'V_mean', 'W_mean']
elif profile == 'T':
    filenames = ['T_mean']
elif profile == 'heatFlux':
    filenames = ['q3_mean', 'Tw_mean']
elif profile == 'TI':
    filenames = ['uu_mean', 'vv_mean', 'ww_mean']

# q3_mean and Tw_mean are for zi calculation, don't add again for the 'heatFlux' profile
filenames += ['q3_mean', 'Tw_mean'] if calc_zi and profile != 'heatFlux' else []
# Reference data directory for U from Churchfield et al. 2012
refDataDir = casedir + '/' + refDataFolder + '/'
if casename in ('ABL_N_L', 'ABL_N_L2'):
    refDataName, caseName2 = 'InflowU_N_L', 'ABL-N-L'
elif casename == 'ABL_N_H':
    refDataName, caseName2 = 'InflowU_N_H', 'ABL-N-H'
elif casename == 'ABL_Uns_L':
    refDataName, caseName2 = 'InflowU_Uns_L', 'ABL-Uns-L'
elif casename == 'ABL_Uns_H':
    refDataName, caseName2 = 'InflowU_Uns_H', 'ABL-Uns-H'
elif casename == 'ABL_N_H_HiSpeed':
    refDataName, caseName2 = '', 'ABL-N-H-HiSpeed'
else:
    refDataName, caseName2 = '', r'N-Low $I$-225^{\circ}-11 m/s'

refDataName += refDataFormat


"""
Read and Process BL Data
"""
# Initialize bl object
bl = BoundaryLayerProfiles(casename=casename, casedir=casedir, force_remerge =force_remerge)
# Read data for given filenames
bl.readPropertyData(filenames=filenames)
# Calculate the temporal average for given starttime and stoptime
bl.calculatePropertyMean(starttime=starttime, stoptime=stoptime)

if calc_zi or profile == 'heatFlux':
    # Calculate zi by finding the minimum z-direction T flux
    # According to statisticsABL.H, q3_mean correspdonds to sgsTempFluxLevelsList.z() from statisticsFace.H,
    # Tw_mean corresponds to
    # tempFluxLevelsList.z()
    minTflux, TfluxLst = 1e20, np.empty_like(bl.hLvls)
    # Go through all height levels
    for i in prange(len(bl.hLvls)):
        Tflux = bl.data['q3_mean_mean'][i] + bl.data['Tw_mean_mean'][i]
        # # However, q3_mean here is from statisticsCell.H
        # # So the following is not feasible
        # if i == 0 or i == len(bl.hLvls) - 1:
        #     Tflux = bl.data['q3_mean_mean'][i]
        # else:
        #     Tflux = bl.data['q3_mean_mean'][i] + 0.5*(bl.data['Tw_mean_mean'][i] + bl.data['Tw_mean_mean'][i - 1])

        # Append Tflux of each height to list for plotting the 'heatFlux' profile
        TfluxLst[i] = Tflux
        if Tflux < minTflux and calc_zi:
            zi, minTflux = bl.hLvls[i], Tflux

figdir = bl.result_dir

# For horizontal U profile
if profile == 'U':
    """
    Plot U BL Profile
    """
    uMean = bl.data['U_mean_mean']
    vMean = bl.data['V_mean_mean']
    wMean = bl.data['W_mean_mean']
    uvMean = np.sqrt(uMean**2 + vMean**2)
    uvwMean = np.sqrt(uMean**2 + vMean**2 + wMean**2)
    # Reference data, 1st column is x, 2nd column is y
    if casename != 'ABL_N_H_HiSpeed':
        refData = readData(refDataName, fileDir=refDataDir, skipRow=0)
        # Since refData is sorted from low x to high x, what we want is from low y to high y
        # Thus sort the 2nd column
        refData = refData[refData[:, 1].argsort()]
        # Add both normalized simulation and reference data to lists to plot
        xList, yList = [uvwMean/Uhub, refData[:, 0]], [bl.hLvls/zi, refData[:, 1]]
        linelabel = (caseName2, 'Churchfield et al.')
    else:
        xList, yList = uvwMean/Uhub, bl.hLvls/zi
        linelabel = None

    # X, Y limit to be inline with Churchfield's data
    xlim, ylim = (0, 2), (0, 1)
    # Initialize figure object
    plot = Plot2D(xList, yList, xlabel=r'$\langle \tilde{U}\rangle /U_0$', ylabel=r'$z/z_i$', figdir=figdir, name=profile, save=save, show=show, xlim=xlim, ylim=ylim, figwidth='1/3')
    plot.initializeFigure()
    # Fill in the rotor swept area
    plot.axes.fill_between(xlim, ((zHub + D/2)/zi,)*2, ((zHub - D/2)/zi,)*2, alpha=0.25, zorder=-1)
    # plot.axes.fill_between(xlim, ((zi + 0.5*invW)/zi,)*2, ((zi - 0.5*invW)/zi,)*2, alpha = fillAlpha,
    #                           facecolor = plot.colors[3], zorder = -2)
    # Plot figure
    plot.plotFigure(linelabel=linelabel)
    plot.finalizeFigure()

    xList2 = [uvMean/Uhub, wMean/Uhub]
    yList2 = [bl.hLvls/zi, bl.hLvls/zi]
    plot2 = Plot2D(xList2, yList2, xlabel=r'$\langle \tilde{\textbf{u}}\rangle /U_0$', ylabel=r'$z/z_i$', figdir=figdir,
                  name='Ucomp', save=save, show=show, xlim=[-.1, 2], ylim=ylim, figwidth='1/3')
    plot2.initializeFigure()
    # Fill in the rotor swept area
    plot2.axes.fill_between([-.1, 2], ((zHub + D/2)/zi,)*2, ((zHub - D/2)/zi,)*2, alpha=0.25, zorder=-1)
    plot2.plotFigure(linelabel=('Horizontal', 'Vertical'))
    plot2.finalizeFigure(xyscale=('linear', 'linear'))

elif profile == 'T':
    """
    Plot T BL Profile
    """
    x = bl.data['T_mean_mean']
    xlim = (x.min()*0.99, x.max()*1.01)
    plot = Plot2D(x, bl.hLvls/zi, xlabel=r'$\langle \tilde{\Theta}\rangle$ [K]', ylabel=r'$z/z_i$ [-]',
                  figdir=figdir, name=profile, save=save, show=show, xlim=xlim, figwidth='1/3')
    plot.initializeFigure()
    plot.plotFigure()

elif profile == 'heatFlux':
    """
    Plot Heat Flux BL Profile
    """
    x = TfluxLst
    # x.min() < 0, thus *1.1
    xlim = (x.min()*1.1, x.max()*1.1)
    plot = Plot2D(x, bl.hLvls, xlabel=r'$\langle \tilde{q_z}\rangle$ [W/m$^2$]',
                  ylabel=r'$z$ [m]',
                  figdir=figdir, name=profile, save=save, show=show, xlim=xlim, figwidth='1/3')
    plot.initializeFigure()
    plot.plotFigure()
    # Limit the number of x ticks since x tick labels are really long
    plt.locator_params(axis='x', nbins=6)

elif profile == 'TI':
    """
    Plot TI BL Profile
    """
    # Reference TI [%] from Churchfield et al. 2012 at rotor bottom, hub, and apex
    if caseName2 == 'ABL-N-L':
        refData = np.array([[5.9, zHub - 0.5*D],
                            [4.9, zHub],
                            [4.4, zHub + 0.5*D]])
    elif caseName2 == 'ABL-N-H':
        refData = np.array([[10.0, zHub - 0.5*D],
                            [8.4, zHub],
                            [7.6, zHub + 0.5*D]])
    # In case not reference data availalble
    else:
        refData = None

    # TI in percentage and normalized by hub height U
    ti = np.sqrt((bl.data['uu_mean_mean'] + bl.data['vv_mean_mean'] + bl.data['ww_mean_mean'])/3.)/Uhub*100.
    xList, yList = ((ti, refData[:, 0]), (bl.hLvls/zi, refData[:, 1]/zi)) if refData is not None else (ti, bl.hLvls/zi)
    xlim = (ti.min()*0.9, ti.max()*1.1)
    plot = Plot2D(xList, yList, xlabel=r'$\langle \tilde{I}\rangle$ [\%]',
                  ylabel=r'$z/z_i$ [-]',
                  figdir=figdir, name=profile, save=save, show=show, xlim=xlim, plot_type=('line', 'scatter'), figwidth='1/3')
    plot.initializeFigure()
    plot.plotFigure(linelabel=(caseName2, 'Churchfield et al.'))
    # # Plot reference data from Churchfield et al. 2012
    # plot.axes[0].scatter(refData[:, 0], refData[:, 1], colo)


# Fill in rotor swept area as well as capping inversion layer for non-normalized plots
if profile != 'U':
    plot.axes.fill_between(xlim, ((zHub + D/2)/zi,)*2, ((zHub - D/2)/zi,)*2, alpha=fillAlpha, zorder=-1)
    # plot.axes.fill_between(xlim, (zi + 0.5*invW,)*2/zi, (zi - 0.5*invW,)*2/zi, alpha=fillAlpha, facecolor=plot.colors[3], zorder=-2)
    plot.finalizeFigure()


# zHub = 90
#
#
# ABL_N_H = InflowProfiles('ABL_N_H', starttime = 14000, stoptime = 18000)
#
# ABL_N_H.getMeanFlowProperty(filenames = ('U_mean', 'V_mean', 'W_mean'), specificZ = zHub)
#
# Vel = np.sqrt(ABL_N_H.propertyDataMean['U_mean']**2 +
#                        ABL_N_H.propertyDataMean['V_mean']**2 +
#                        ABL_N_H.propertyDataMean['W_mean']**2)
#
# # VelHub = np.sqrt(ABL_N_H.propertyMeanSpecificZ['U_mean']**2 +
# #                        ABL_N_H.propertyMeanSpecificZ['V_mean']**2 +
# #                        ABL_N_H.propertyMeanSpecificZ['W_mean']**2)
# VelHub = 8.
#
# dataChurchfield = readData('ABL_InflowU_Churchfield.csv', fileDir = './Churchfield', skipRow = 2)
# # UUhub_NL, zzi_NL = dataChurchfield['X'], dataChurchfield['Y']
# # UUhub_NH, zzi_NH = dataChurchfield['X.1'], dataChurchfield['Y.1']
# # UUhub_UL, zzi_UL = dataChurchfield['X.2'], dataChurchfield['Y.2']
# # UUhub_UH, zzi_UH = dataChurchfield['X.3'], dataChurchfield['Y.3']
#
# UUhub_NL, zzi_NL = dataChurchfield[:, 0], dataChurchfield[:, 1]
# # UUhub_NH, zzi_NH = dataChurchfield['X.1'], dataChurchfield['Y.1']
# # UUhub_UL, zzi_UL = dataChurchfield['X.2'], dataChurchfield['Y.2']
# # UUhub_UH, zzi_UH = dataChurchfield['X.3'], dataChurchfield['Y.3']
#
#
# ABL_N_H.getMeanFlowProperty(filenames = 'T_mean')
#
# ABL_N_H.getMeanFlowProperty(filenames = 'q3_mean')
#
# ABL_N_H.getMeanFlowProperty(filenames = ('uu_mean', 'vv_mean', 'ww_mean'))
#
# q3_mean = abs(ABL_N_H.propertyDataMean['q3_mean'])
#
# q3_mean_minLoc = np.argmin(q3_mean)
# # zi = ABL_N_H.z[q3_mean_minLoc]
#
# TI = np.divide(np.sqrt(ABL_N_H.propertyDataMean['uu_mean'] +
#                        ABL_N_H.propertyDataMean['vv_mean'] +
#                        ABL_N_H.propertyDataMean['ww_mean'])/3, Vel)
#
# zLocInversionLow = np.where(ABL_N_H.z >= 700)[0][0] - 1
# zLocInversionHi = np.where(ABL_N_H.z >= 800)[0][0]
#
# gapTIold = 0
# TIinversion = TI[zLocInversionLow:zLocInversionHi]
# for i, val in enumerate(TIinversion[:-1]):
#     if abs(TIinversion[i + 1] - TIinversion[i]) > gapTIold:
#         gapTIold = abs(TIinversion[i + 1] - TIinversion[i])
#         ziLoc = i + zLocInversionLow
#
#
# zi = (ABL_N_H.z[ziLoc] + ABL_N_H.z[ziLoc + 1])/2






# Plot2D(X = Vel/VelHub, Y = ABL_N_H.z/zi, figdir = './', xlabel = r'$<U>/U_{Hub}$',
#        ylabel = r'$z/z_i$',
# fileName = ABL_N_H.casename + '_U', xlim = [0, 2], ylim = [0, 1])
#
# Plot2D(X = ABL_N_H.propertyDataMean['T_mean'], Y = ABL_N_H.z, figdir = './', xlabel = r'$<T>$',
#        fileName = ABL_N_H.casename + '_T')

# Plot2D(plot = 'line', X = ABL_N_H.propertyDataMean['q3_mean'], Y = ABL_N_H.z, figdir = './', \
#                                                                                                xlabel = r'$<q_3>$',
#        fileName = ABL_N_H.casename + '_q3')

# Plot2D(plot = 'line', X = TI, Y = ABL_N_H.z, figdir = './', \
#                                                                                                xlabel = r'$<TI>$',
#        ylabel = 'z',
#        fileName = ABL_N_H.casename + '_TI')


# import matplotlib.pyplot as plt

# plt.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
# plt.show()
# Plot2D(X = np.linspace(0, 100, 100), Y = np.linspace(0, 100, 100))

# test.getTimesAndIndices()
#
# test.velocity()

#
# test.get_Zi()
#
#
# test.startTimeReal
#
#
# ensembleFolderPath = test.mergeTimeDirectories()
#
# startTimeReal, stopTimeReal, iStart, iStop = test.getTimesAndIndices(ensembleFolderPath)
#
# dataUmean, dataVmean, dataWmean = \
#     test.velocity(ensembleFolderPath,
#                   iStart = iStart, iStop = iStop, startTimeReal = startTimeReal, stopTimeReal = stopTimeReal)

