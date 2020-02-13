import numpy as np
import os
from PlottingTool import Plot2D
import PostProcess_EnergySpectrum as PPES
import time
from Utilities import timer
import pickle
from Utilities import readData


"""
User Inputs
"""
case = 'ABL_N_H'  # 'Doekemeijer', 'ABL_N_L2'
# caseDir = '/media/yluan/Toshiba External Drive/'
caseDir = '/media/yluan/'
# If slice time is 'auto', use the 1st time directory in slice folder
sliceTime = 'auto'  # 'auto', '<time>'
# Whether leave E33 out of Eii, default True by Churchfield
horizontalEii = True
sliceFolder, resultFolder = 'Slices', 'Result'
sliceName = 'U_hubHeight_Slice.raw'
refDataFolder, refDataFormat = 'Churchfield', '.csv'
# Cell size in x, y/z directions, has to be a tuple
cellSizes2D = (10., 10.)
# Domain size, to normalize Kr wavenumber
L = 3000.
# Molecular viscosity for the Kolmogorov -5/3 model
nu = 1e-5  # [m^2/s]
# Large eddy length scale, for Kolmogorov -5/3 model
Lt = 250  # [CAUTION]


"""
Plot Settings
"""
show, save = False, True
xlim, ylim = (1e-3, 1), (1e-6, 1)
xLabels = '$K_r$ [1/m]'
# Alpha of fill-in of SFS region
fillAlpha = 0.25


"""
Process User Inputs
"""
caseFullPath = caseDir + '/' + case + '/' + sliceFolder + '/'
# If slice time is 'auto', select the 1st time directory
sliceTime = os.listdir(caseFullPath)[0] if sliceTime == 'auto' else sliceTime
resultPath = caseFullPath + resultFolder + '/' + sliceTime + '/'
if case == 'ABL_N_H':
    label, E12refName, E33refName = 'ABL-N-H', 'E12_N_H', 'E33_N_H'
elif case in ('ABL_N_L', 'ABL_N_L2'):
    label, E12refName, E33refName = 'ABL-N-L', 'E12_N_L', 'E33_N_L'
elif case == 'Doekemeijer':
    label, E12refName, E33refName = r'N, $U_{\mathrm{hub}} = 8$ m/s, $z_0 = 0.15$ m', 'E12_N_H', 'E33_N_H'
else:
    label, E12refName, E33refName = 'ABL-N-H-HiSpeed', 'E12_N_H', 'E33_N_H'

E12refName += refDataFormat
E33refName += refDataFormat
refDataDir = caseDir + '/' + refDataFolder
# Read reference data, 1st column is x, 2nd column is y
if case != 'ABL_N_H_HiSpeed':
    E12ref, E33ref = readData(E12refName, fileDir=refDataDir), readData(E33refName, fileDir=refDataDir)
    # In case data is not sorted from low x to high x
    E12ref, E33ref = E12ref[E12ref[:, 0].argsort()], E33ref[E33ref[:, 0].argsort()]


"""
Calculate Energy Spectrum
"""
@timer
# jit(parallel = True) gives unknown error
# prange has to be used with parallel = True
def getPlanarEnergySpectrum(u2D, v2D, w2D, L, cellSizes2D, horizontalEii=False):
    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
    # TKE calculated form physical space
    TKE = 0.5*np.sum(uRes2D**2 + vRes2D**2 + wRes2D**2)
    # Number of samples in x and y/z, columns correspond to x
    nPtX, nPtY = uRes2D.shape[1], uRes2D.shape[0]
    # 2D DFT, no normalization (will be done manually below)
    uResFft, vResFft, wResFft = np.fft.fft2(uRes2D, axes = (0, 1), norm = None), \
                                np.fft.fft2(vRes2D, axes = (0, 1), norm = None), \
                                np.fft.fft2(wRes2D, axes = (0, 1), norm = None)
    # Normalization by N^(dimension)
    uResFft /= (nPtX*nPtY)
    vResFft /= (nPtX*nPtY)
    wResFft /= (nPtX*nPtY)

    # Corresponding frequency in x and y/z directions, expressed in cycles/m
    # Kx corresponds to k/n in np.fft.fft documentation, where
    # k = 0, ..., (n - 1)/2; n is number of samples
    # Number of columns are number of x
    # d is sample spacing, which should be equidistant,
    # in this case, cell size in x and y/z respectively
    Kx, Ky = np.fft.fftfreq(nPtX, d = cellSizes2D[0]), np.fft.fftfreq(nPtY, d = cellSizes2D[1])
    # Kx and Ky/Kz is defined as 2n*pi/L, while the K in np.fft.fftn() is simply n/L, n in [1, N]
    # Thus scale old Kx, Ky/Kz by 2pi and 2D meshgrid treatment
    Kx *= 2*np.pi
    Ky *= 2*np.pi
    Kx, Ky = np.meshgrid(Kx, Ky)
    # Before calculating (cross-)correlation, add arrays to 2 tuples
    U1ResFft = (uResFft, uResFft, uResFft, vResFft, vResFft, wResFft)
    U2ResFft = (uResFft, vResFft, wResFft, vResFft, wResFft, wResFft)
    # Initialize 2D Rij in spectral space
    RijFft = np.empty((nPtY, nPtX, 6), dtype = np.complex128)
    # Go through each component of RijFft
    # The 6 components are 11, 12, 13,
    # 22, 23,
    # 33
    for i in range(6):
        # Perform the 2-point (cross-)correlation
        RijFft[:, :, i] = np.multiply(U1ResFft[i], np.conj(U2ResFft[i]))

    # Trace of 2-point correlations
    # If decompose Rii to horizontal Rii and R33
    if horizontalEii:
        RiiFft = RijFft[:, :, 0] + RijFft[:, :, 3]
    else:
        RiiFft = RijFft[:, :, 0] + RijFft[:, :, 3] + RijFft[:, :, 5]

    # Original resultant Kr
    KrOld = np.sqrt(Kx**2 + Ky**2)
    # New proposed Kr for E spectrum, same number of points as x
    Kr0 = 2*np.pi/L
    Kr = Kr0*np.linspace(1, nPtX, nPtX)
    # Initialize Eij combined from 2D to 1D
    # Eij[i] is 0.5ui'uj' = 0.5sum(Rij of equal Kr[i])
    Eij = np.empty((len(Kr), 6), dtype = np.complex128)
    Eii = np.empty_like(Kr, dtype = np.complex128)
    # Occurrence when KrOld is close to each Kr[i]
    occurrence = np.empty(len(Kr))
    # Go through each proposed Kr
    # Integrate Rij where KrOld lies between Kr0*[(i + 1) - 0.5, (i + 1) + 0.5)
    # This is possible since RijFft and KrOld has matching 2D indices
    for i in range(len(Kr)):
        occurrence[i] = len(KrOld[(KrOld >= Kr0*(i + 1 - 0.5)) & (KrOld < Kr0*(i + 1 + 0.5))])
        # For Eij, go through all 6 components
        for j in range(6):
            Eij[i, j] = 0.5*np.sum(RijFft[:, :, j][(KrOld >= Kr0*(i + 1 - 0.5)) & (KrOld < Kr0*(i + 1 + 0.5))])

        Eii[i] = 0.5*np.sum(RiiFft[(KrOld >= Kr0*(i + 1 - 0.5)) & (KrOld < Kr0*(i + 1 + 0.5))])

    # # If done right, TKE from energy spectrum should equal TKE in physical space
    # TKE_k = sum(Eii)*nPtX*nPtY

    # # Optional equal Kr histogram
    # plt.figure('Histogram')
    # plt.hist(occurrence)

    return RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr, TKE


t0 = time.time()
# Read 2D mesh and velocity
x2D, y2D, z2D, U2D, u2D, v2D, w2D = PPES.readStructuredSliceData(sliceName=sliceName, case=case, caseDir=caseDir, time=sliceTime)
t1 = time.time()
print('\nFinished readSliceRawData in {:.4f} s'.format(t1 - t0))

# Calculate 2-point (cross-)correlation and energy spectrum density and corresponding wave number
t0 = time.time()
RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr, TKE = PPES.getPlanarEnergySpectrum(u2D, v2D, w2D, L=L, cellSizes2D=cellSizes2D, horizontalEii=horizontalEii)
Eii, Eij = abs(Eii), abs(Eij)
t1 = time.time()
print('\nFinished getPlanarEnergySpectrum in {:.4f} s'.format(t1 - t0))


"""
Prepare the Kolmogorov -5/3 Model
"""
# Kolmogorov dimensional analysis: Lt ~ TKE^(3/2)/epsilon
# Lt is characteristic length scale of large eddies
# # epsilon is dissipation rate
# # epsilon = Lt/(TKE**(3/2.))
# epsilon = 0.
# for i in prange(1, len(Kr)):
#     # Scale Eii back and normalize it in the end
#     # epsilon += 2*nu*Kr[i]**2*Eii[i]*norm*(Kr[i] - Kr[i - 1])
#     epsilon += 2*nu*(Kr[i]/krFactor)**2*Eii[i]*norm[i]*(Kr[i] - Kr[i - 1])/krFactor

# Kolmogorov -5/3 model
# # E(Kr) = c_k*epsilon^(2/3)*Kr^(-5/3)
# # c_k = 1.5 generally
# E_Kolmo = 1.5*epsilon**(2/3.)*(Kr/krFactor)**(-5/3)/norm
# E_Kolmo = 1.5*epsilon**(2/3.)*(Kr)**(-5/3)/norm
# E_Kolmo = np.divide(1.5*epsilon**(2/3.)*(Kr/krFactor)**(-5/3), norm)
# Kolmogorv E is already normalized by Lt*Kr
E_Kolmo = 1.5*(Lt*Kr)**(-5/3)


"""
Plot the Energy Spectrums of Eii, E12, E33
"""
# E to be plotted into a tuple
E = (Eii, Eij[:, -1])
# Figure order is Eii, E33
figNames = (case + '_Eii_hor', case + '_E33')

yLabels = (r'$E_{11} + E_{22}$ [m$^3$/s$^2$]', r'$E_{33}$ [m$^3$/s$^2$]')

# Go through Eii, E33 and plot each of them
for i in range(len(E)):
    # If plotting horizontal Eii
    if i != len(E) - 1:
        if case != 'ABL_N_H_HiSpeed':
            xList, yList = (Kr, E12ref[:, 0], Kr), (E[i], E12ref[:, 1], E_Kolmo)
            plot_label = (label, 'Churchfield et al.', 'Kolmogorov model')
        else:
            xList, yList = (Kr, Kr), (E[i], E_Kolmo)
            plot_label = (label, 'Kolmogorov model')

    # Else if plotting E33
    else:
        if case != 'ABL_N_H_HiSpeed':
            xList, yList = (Kr, E33ref[:, 0], Kr), (E[i], E33ref[:, 1], E_Kolmo)
            plot_label = (label, 'Churchfield et al.', 'Kolmogorov model')
        else:
            xList, yList = (Kr, Kr), (E[i], E_Kolmo)
            plot_label = (label, 'Kolmogorov model')

    # Initialize figure
    plot = Plot2D(xList, yList, xlabel=xLabels, ylabel=yLabels[i], name=figNames[i], save=save, show=show, xlim=xlim, ylim=ylim, figdir=resultPath, figwidth='1/3')
    plot.initializeFigure()
    plot.plotFigure(linelabel=plot_label)
    plot.axes.fill_between((1./cellSizes2D[0], xlim[1]), ylim[0], ylim[1], alpha=fillAlpha, facecolor=plot.gray, zorder=-1)
    plot.finalizeFigure(xyscale=('log', 'log'))




# [DEPRECATED]
"""
Binning Operation
"""
# # Kr manipulation
# # Kr *= krFactor
# # Mean cell size, used for the fill-in plot of the SFS region
# cellSize = np.mean(cellSizes)
# # Target Kr after binning
# Kr_binned = np.logspace(-3, 0, nBin)
# # Kr_binned = np.linspace(Kr.min(), Kr.max(), 1000)
# # Indices of which bin each Kr would fall in
# iBin = np.digitize(Kr, Kr_binned)
# # Get all unique iBin values
# iBin_uniq = np.unique(iBin)
# # Go through every index of targeted bins
# for i in prange(nBin):
#     # If such index not found in iBin, it means the bins in that region is too refined for binning old values
#     # E.g. Kr = [1, 2, 3],
#     # Kr_binned = [1, 1.5, 2, 2.5, 3],
#     # no Kr to put in Kr_binned of 1.5 or 2.5
#     # Make those over-flown Kr bin NaN
#     if i not in iBin_uniq:
#         Kr_binned[i] = np.nan
#
# # Then remove those NaN entries
# Kr_binned = Kr_binned[~np.isnan(Kr_binned)]
# # Refresh the bin indices that each Kr should fall in
# iBin = np.digitize(Kr, Kr_binned)
# # Then, for all E that fall in a bin, get the average
# # Go through every Eij column, 6 in total, lastly, do it for Eii too
# Eij_binned_0 = np.array([np.mean(Eij[iBin == i, 0]) for i in range(len(Kr_binned))])
# Eij_binned = np.empty((Eij_binned_0.shape[0], 6))
# Eij_binned[:, 0] = Eij_binned_0
# for j in prange(1, 7):
#     # For Eij
#     if j != 6:
#         Eij_binned[:, j] = np.array([np.mean(Eij[iBin == i, j]) for i in range(len(Kr_binned))])
#     # For Eii
#     else:
#         Eii_binned = np.array([np.mean(Eii[iBin == i]) for i in range(len(Kr_binned))])


@timer
def getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes, type = 'decomposed'):
    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()

    print(uRes2D.mean(), vRes2D.mean(), wRes2D.mean())

    #    # Normalized
    #    uRes2D, vRes2D, wRes2D = (u2D - u2D.mean())/u2D.mean(), (v2D - v2D.mean())/v2D.mean(), (w2D - w2D.mean(
    #    ))/w2D.mean()

    # Perform 2D FFT
    uResFft = np.fft.fft2(uRes2D, axes = (0, 1))
    vResFft = np.fft.fft2(vRes2D, axes = (0, 1))
    wResFft = np.fft.fft2(wRes2D, axes = (0, 1))
    # Shift FFT results so that 0-frequency component is in the center of the spectrum
    uResFft, vResFft, wResFft = np.fft.fftshift(uResFft), np.fft.fftshift(vResFft), np.fft.fftshift(wResFft)

    nX, nY = uRes2D.shape[1], uRes2D.shape[0]
    # Frequencies are number of rows/columns with a unit of cellSize meters
    # Note these are wavelength vector (Kx, Ky)
    freqX, freqY = np.fft.fftfreq(nX, d = cellSizes[0]), np.fft.fftfreq(nY, d = cellSizes[1])
    # Also shift corresponding frequencies
    freqX, freqY = np.fft.fftshift(freqX), np.fft.fftshift(freqY)

    # return uResFft, vResFft, wResFft, freqX, freqY

    # Calculate energy density Eii(Kx, Ky) (per d(Kx, Ky))
    # If 'decomposed' type, calculate E of horizontal velocity and vertical separately
    if type == 'decomposed':
        # Eii(Kx, Ky) = 0.5(|uResFft(Kx, Ky)^2| + ...)
        # u*np.conj(u) equals |u|^2
        # abs() to get the real part
        Eii = 0.5*np.abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft))
        #        # Not sure about this
        #        Eij = 0.5*np.abs(uResFft*np.conj(vResFft) + vResFft*np.conj(uResFft))
        # Vertical E separate from the horizontal one
        E33 = 0.5*np.abs(wResFft*np.conj(wResFft))
    else:
        Eii = 0.5*np.abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft) + wResFft*np.conj(wResFft))
        # Dummy array for vertical E33 and Eij
        Eij, E33 = np.empty((wResFft.shape[0], wResFft.shape[1])), np.empty((wResFft.shape[0], wResFft.shape[1]))

    # No Eij yet

    # Convert (Kx, Ky) to Kr for 1D energy spectrum result/plot
    # First, get all Kr[i_r] = sqrt(Kx^2 + Ky^2) and its corresponding Eii(Kr)
    Kr = np.empty(len(freqX)*len(freqY))
    Eii_r, E33_r = np.empty(len(freqX)*len(freqY)), np.empty(len(freqX)*len(freqY))
    i_r = 0
    for iX in range(len(freqX)):
        for iY in range(len(freqY)):
            Kr[i_r] = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
            Eii_r[i_r], E33_r[i_r] = Eii[iY, iX], E33[iY, iX]
            i_r += 1

    # Then, sort Kr from low to high and sort Eii(Kr) accordingly
    KrSorted = np.sort(Kr)
    sortIdx = np.argsort(Kr)
    EiiSorted, E33Sorted = Eii_r[sortIdx], E33_r[sortIdx]

    # Lastly, find all Eii, Eij, E33 of equal Kr and line-integrate over co-centric Kr (would have been sphere for
    # slice not slice)
    E, Evert, KrFinal = [], [], []
    i = 0
    # Go through all Kr from low to high
    while i < len(KrSorted):
        # Remember E of current Kr[i]
        Eii_repeated, E33_repeated = EiiSorted[i], E33Sorted[i]
        match = 0.
        # Go through the rest of Kr other than Kr[i]
        for j in range(i + 1, len(KrSorted)):
            # If the other Kr[j] matches current Kr[i]
            if KrSorted[j] == KrSorted[i]:
                Eii_repeated += EiiSorted[j]
                E33_repeated += E33Sorted[j]
                match += 1.

        # If multiple matches, i.e. if multiple E in the same cocentric ring, then get the average E
        if match > 0:
            Eii_repeated /= (match + 1)
            E33_repeated /= (match + 1)

        #        E.append(Eii_repeated*2*np.pi*KrSorted[i])
        #        Evert.append(E33_repeated*2*np.pi*KrSorted[i])
        E.append(Eii_repeated)
        Evert.append(E33_repeated)
        KrFinal.append(KrSorted[i])
        i += int(match) + 1

    print('\nEnergy spectrum calculated')
    return np.array(E), np.array(Evert), np.array(KrFinal)


# plt.figure('Eii')
# plt.ylabel("Eii")
# plt.xlabel("Frequency 10Kr [cycle/m?]")
# # plt.bar(f[:N//2], np.abs(uResFft)[:N//2]*1./N, width = 1.5)  # 1 / N is a normalization factor
# # Kr manipulation here!
# plt.loglog(Kr, np.abs(Eii)/len(Eii))
# plt.loglog(Kr, E_Kolmo)
# plt.xlim(1e-3, 1)
# plt.ylim(1e-6, 1)
# plt.show()
#
# plt.figure('E12')
# plt.ylabel("E12")
# plt.xlabel("Frequency 10Kr [cycle/m?]")
# # plt.bar(f[:N//2], np.abs(uResFft)[:N//2]*1./N, width = 1.5)  # 1 / N is a normalization factor
# # E12 is second column of Eij
# # Kr manipulation here!
# plt.loglog(Kr, np.abs(Eij[:, 1])/len(Eij))
# plt.loglog(Kr, E_Kolmo)
# plt.xlim(1e-3, 1)
# plt.ylim(1e-6, 1)
# plt.show()
#
# plt.figure('E33')
# plt.ylabel("E33")
# plt.xlabel("Frequency 10Kr [cycle/m?]")
# # plt.bar(f[:N//2], np.abs(uResFft)[:N//2]*1./N, width = 1.5)  # 1 / N is a normalization factor
# # E33 is last column of Eij
# # Kr manipulation here!
# plt.loglog(Kr, np.abs(Eij[:, -1])/len(Eij))
# plt.loglog(Kr, E_Kolmo)
# plt.xlim(1e-3, 1)
# plt.ylim(1e-6, 1)
# plt.show()



# E, Evert, Kr = getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = [10., 10.])
#
# pickle.dump(E, open(resultPath + 'E.p', 'wb'))
# pickle.dump(Evert, open(resultPath + 'Evert.p', 'wb'))
# pickle.dump(Kr, open(resultPath + 'Kr.p', 'wb'))
#
# """
# Plot Energy Spectrum
# """
# plt.figure('horizontal')
# plt.loglog(Kr, E)
# # plt.loglog(Kr, Kr**(-5/3.)*(9e-3)**(2/3.)*1.5)
# plt.xlim(1e-3, 1)
# plt.tight_layout()
#
# plt.figure('vertical')
# plt.loglog(Kr, Evert)
# # plt.loglog(Kr, Kr**(-5/3.)*(9e-3)**(2/3.)*1.5)
# plt.xlim(1e-3, 1)
# plt.tight_layout()


# def readSliceRawData(field, case = 'ABL_N_H/Slices', caseDir = './', time = None, skipHeader = 2):
#     timePath = caseDir + '/' + case + '/'
#     if time is None:
#         time = os.listdir(timePath)
#         try:
#             time.remove('Result')
#         except:
#             pass
#
#         time = time[0]
#
#     fieldFullPath = timePath + str(time) + '/' + field
#     data = np.genfromtxt(fieldFullPath, skip_header = skipHeader)
#
#     # 1D array
#     x, y, z = data[:, 0], data[:, 1], data[:, 2]
#
#     # Mesh size in x
#     valOld = x[0]
#     for i, val in enumerate(x[1:]):
#         if val < valOld:
#             nPtX = i + 1
#             break
#
#         valOld = val
#
#     x2D, y2D, z2D = x.reshape((-1, nPtX)), y.reshape((-1, nPtX)), z.reshape((-1, nPtX))
#
#     if data.shape[1] == 6:
#         u, v, w = data[:, 3], data[:, 4], data[:, 5]
#         scalarField = np.zeros((data.shape[0], 1))
#         for i, row in enumerate(data):
#             scalarField[i] = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2)
#
#     else:
#         u, v, w = np.zeros((data.shape[1], 1)),np.zeros((data.shape[1], 1)), np.zeros((data.shape[1], 1))
#         scalarField = data[:, 3]
#
#     u2D, v2D, w2D = u.reshape((-1, nPtX)), v.reshape((-1, nPtX)), w.reshape((-1, nPtX))
#     scalarField2D = scalarField.reshape((-1, nPtX))
#
#     print('\nSlice raw data read')
#     return x2D, y2D, z2D, scalarField2D, u2D, v2D, w2D
#
#
#
# x2D, y2D, z2D, U2D, u2D, v2D, w2D = readSliceRawData('U_hubHeight_Slice.raw')
#
#
# def getEnergySpectrum(u2D, v2D, w2D):
#     uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
#
#     uResFft = np.fft.fft2(uRes2D)
#     vResFft = np.fft.fft2(vRes2D)
#     wResFft = np.fft.fft2(wRes2D)
#     uResFft, vResFft, wResFft = np.fft.fftshift(uResFft), np.fft.fftshift(vResFft), np.fft.fftshift(wResFft)
#
#     nX, nY = uRes2D.shape[1], uRes2D.shape[0]
#     freqX, freqY = np.fft.fftfreq(nX, d = 10.), np.fft.fftfreq(nY, d = 10.)
#     freqX, freqY = np.fft.fftshift(freqX), np.fft.fftshift(freqY)
#
#     Eii = abs(uResFft*np.conj(uResFft) + vResFft*np.conj(vResFft))
#
#     Kr = np.zeros((len(freqX)*len(freqY), 1))
#     EiiR = np.zeros((len(freqX)*len(freqY), 1))
#     iR = 0
#     for iX in range(len(freqX)):
#         for iY in range(len(freqY)):
#             # Kcurrent = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
#             # if Kcurrent in Kr:
#
#             Kr[iR] = np.sqrt(freqX[iX]**2 + freqY[iY]**2)
#             EiiR[iR] = Eii[iY, iX]
#             iR += 1
#
#     KrSorted = np.sort(Kr, axis = 0)
#     sortIdx = np.argsort(Kr, axis = 0).ravel()
#     EiiSorted = EiiR[sortIdx]
#
#     EiiReduced, Kreduced = [], []
#     i = 0
#     while i < len(KrSorted):
#         EiiRrepeated = EiiSorted[i]
#         anyMatch = False
#         for j in range(i + 1, len(KrSorted)):
#             if KrSorted[j] == KrSorted[i]:
#                 EiiRrepeated += EiiSorted[j]
#                 skip = j
#                 anyMatch = True
#
#         if not anyMatch:
#             skip = i
#
#         EiiReduced.append(EiiRrepeated)
#         Kreduced.append(KrSorted[i])
#         i = skip + 1
#
#     return EiiReduced, Kreduced
#
#
#
#
# # # Shift freqs all to non-negative
# # kX, kY = 2*np.pi*(freqX - freqX.min()), 2*np.pi*(freqY - freqY.min())
# #
# # krOld = 0
# # E, kr = np.zeros((uResFft.shape[0], 1)), np.zeros((uResFft.shape[0], 1))
# # for i in range(uResFft.shape[0]):
# #     kr[i] = np.sqrt(kX[i]**2 + kY[i]**2)
# #     dk = abs(krOld - kr[i])
# #     # This should depend on K
# #     eii = float(uResFft[i, i]*np.conj(uResFft[i, i])) + float(vResFft[i, i]*np.conj(vResFft[i, i]))
# #     E[i] = eii/2.
# #
# #     krOld = kr[i]
#
# plt.figure('uv')
# plt.loglog(Kreduced, EiiReduced)


# import numpy as np
# import vtk
# from vtk.numpy_interface import dataset_adapter as dsa
# from vtk.util import numpy_support as VN
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import meshio
#
# # mesh = meshio.read('./ABL_N_H/Slices/20060.9038025/U_slice_horizontal_2.vtk')
# reader = vtk.vtkPolyDataReader()
# reader.SetFileName('./ABL_N_H/Slices/20060.9038025/U_slice_horizontal_2.vtk')
# # reader.ReadAllVectorsOn()
# # reader.ReadAllScalarsOn()
# reader.Update()
#
# polydata = reader.GetOutput()
#
# cellArr = dsa.WrapDataObject(polydata).Polygons
#
# # ptLst = np.zeros((1, 3))
# # for i in range(polydata.GetNumberOfCells()):
# #    pts = polydata.GetCell(i).GetPoints()
# #    # cells = polydata.GetCell(i)
# #    np_pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
# #    ptLst = np.vstack((ptLst, np_pts))
# #    print(np_pts)
#
#
# # # dim = data.GetDimensions()
# # dim = (0, 0, 0)
# # vec = list(dim)
# # vec = [i-1 for i in dim]
# # vec.append(3)
# #
# # u = VN.vtk_to_numpy(polydata.GetCellData().GetArray('U'))
# # # uHor =
# # # b = VN.vtk_to_numpy(data.GetCellData().GetArray('POINTS'))
# #
# # # u = u.reshape((300, 300), order='F')
# # # b = b.reshape(vec,order='F')
# #
# # x = np.zeros(data.GetNumberOfPoints())
# # y = np.zeros(data.GetNumberOfPoints())
# # z = np.zeros(data.GetNumberOfPoints())
# #
# # # xMesh, yMesh = np.meshgrid(x, y)
# #
# # for i in range(data.GetNumberOfPoints()):
# #     x[i],y[i],z[i] = data.GetPoint(i)
# #
# # # Sort xy. First based on x, then y (x doesn't move)
# # xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
# #
# #
# # p2c = vtk.vtkPointDataToCellData()
# # p2c.SetInputConnection(reader.GetOutputPort())
# # p2c.Update()
# #
# # a = p2c.GetOutput()
# # b = VN.vtk_to_numpy(a.GetCellData().GetArray('POINTS'))
# # # iterate over blocks and copy in the result
# #
# # iter=dsa.MultiCompositeDataIterator([p2c.GetOutputDataObject(0), output])
# #
# # for  in_block,  output_block in iter:
# #
# #      output_block.GetCellData().AddArray(in_block.VTKObject.GetCellData().GetArray('DISPL'))
# #
# #
# #
# #
# # a = p2c
# # warp = vtk.vtkWarpVector()
# # b = warp.SetInputConnection(p2c.GetOutputPort())
# #
# # # for i, row in enumerate(xy):
# # #     if row[0] == 0:
# #
# #
# #
# # # plt.figure('x')
# # # plt.scatter(np.arange(0, x.shape[0]), x)
# # #
# # # plt.figure('y')
# # # plt.scatter(np.arange(0, x.shape[0]), y)
# #
# #
# # # x = x.reshape((301, 301), order='F')
# # # y = y.reshape((301, 301),order='F')
# # # z = z.reshape((301, 301),order='F')
# # #
# # # plt.figure()
# # # plt.contour(x[:-1, :-1], y[:-1, :-1], u)
