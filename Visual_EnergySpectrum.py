import numpy as np
import os
from PlottingTool import Plot2D
import PostProcess_EnergySpectrum
import time
from Utilities import timer
from numba import jit, prange
import pickle
from Utilities import readData

"""
User Inputs
"""
caseDir, case = 'J:', 'ABL_N_H'
# caseDir = '/media/yluan/Toshiba External Drive/'
caseDir = '/media/yluan/1'
# If slice time is 'auto', use the 1st time directory in slice folder
sliceTime = 'auto'  # 'auto', '<time>'
# Whether to merge Kx and Ky into Kr
mergeXY = False
sliceFolder, resultFolder = 'Slices', 'Result'
sliceName = 'U_hubHeight_Slice.raw'
refDataFolder, refDataFormat = 'Churchfield', '.csv'
# Cell size in x, y, and z directions
cellSizes = (10., 10., 10.)
# Whether to normalize the DFT by sqrt(Nsample)
fftNorm = None  # 'ortho', None
# Whether to normalize E(Kr), if so, normalize every E(Kr) by (Kr*Lt*nPt_r)
# Default is True, as done by Churchfield
plotNorm = True
# Molecular viscosity for the Kolmogorov -5/3 model
nu = 1e-5  # [m^2/s]
# Large eddy length scale, for Kolmogorov -5/3 model
Lt = 250.  # [CAUTION]
# Kr manipulation
# Whether to multiply Kr by krFactor
# This only shifts the x-axis of the energy spectrum
# From literature, Kr is defined as 2pi*K, where K is derived from np.fft.fftfreq()
krFactor = 1  # [CAUTION]



"""
Plot Settings
"""
# Number of bins
nBin = 150
show, save = False, True
xLim, yLim = (1e-3, 1), (1e-6, 1)
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
    label, E12refName, E33refName = 'ABL', 'E12_N_H', 'E33_N_H'

E12refName += refDataFormat
E33refName += refDataFormat
refDataDir = caseDir + '/' + refDataFolder
# Read reference data, 1st column is x, 2nd column is y
E12ref, E33ref = readData(E12refName, fileDir = refDataDir), readData(E33refName, fileDir = refDataDir)
# In case data is not sorted from low x to high x
E12ref, E33ref = E12ref[E12ref[:, 0].argsort()], E33ref[E33ref[:, 0].argsort()]


"""
Calculate Energy Spectrum
"""
@timer
# jit(parallel = Truesignal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)) gives unknown error
@jit(fastmath = True)
def getPlanarEnergySpectrum(u2D, v2D, w2D, cellSizes):
    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
    # Before calculating (cross-)correlation, add arrays to 2 tuples
    U1Res2D = (uRes2D, uRes2D, uRes2D, vRes2D, vRes2D, wRes2D)
    U2Res2D = (uRes2D, vRes2D, wRes2D, vRes2D, wRes2D, wRes2D)
    # Number of samples in x and y
    nPtX, nPtY = uRes2D.shape[1], uRes2D.shape[0]
    # Take the lower bound integer of half of nPt, applies to both even and odd nPtX/nPtY
    nPtKx, nPtKy = nPtX//2, nPtY//2
    # 1 point (cross-)correlation of velocity fluctuations in spatial domain
    Rij = np.empty((nPtY, nPtX, 6))
    # 2D DFT of Rij will be half of Rij sizes since the other half are just conjugates that's not interesting, given that Rij(x, y) are all real
    RijFft = np.empty((nPtKy, nPtKx, 6), dtype = np.complex128)
    # The 6 components are 11, 12, 13,
    # 22, 23,
    # 33,
    for i in prange(6):
        # Perform the 1-point (cross-)correlation
        Rij[:, :, i] = np.multiply(U1Res2D[i], U2Res2D[i])
        # Take only the 1st half in each direction as the 2nd half is not interesting
        # RijFft[0, 0] is the sum of Rij by definition
        RijFft[:, :, i] = (np.fft.fft2(Rij[:, :, i], axes = (0, 1)))[:nPtKy, :nPtKx]

    # Trace of 1-point correlations, equivalent to 2TKE
    Rii = Rij[:, :, 0] + Rij[:, :, 3] + Rij[:, :, 5]
    # 2D DFT of Rii
    RiiFft = np.fft.fft2(Rii, axes = (0, 1))[:nPtKy, :nPtKx]
    # RiiFft[0, 0] is the sum of Rii by definition, which is 2TKE
    TKE = np.abs(0.5*RiiFft[0, 0])
    # Corresponding frequency in x and y directions, expressed in cycles/m
    # Number of columns are number of x
    # d is sample spacing, which should be equidistant,
    # in this case, cell size in x and y respectively
    # Again, only take the first half, the positive frequencies
    Kx, Ky = np.fft.fftfreq(nPtX, d = cellSizes[0])[:nPtKx], np.fft.fftfreq(nPtY, d = cellSizes[1])[:nPtKy]
    # Resultant K after combining Kx and Ky
    Kr, iKxy = np.empty(nPtKx*nPtKy), []
    cnt = 0
    # Go through each Kx, then Ky
    for i in prange(len(Kx)):
        for j in range(len(Ky)):
            # Take the resultant
            Kr[cnt] = np.sqrt(Kx[i]**2 + Ky[j]**2)
            # To access i/j, use iKxy[row][col] format since iKxy is a 2D list instead of 2D np.ndarray
            iKxy.append((i, j))
            cnt += 1

    iKxy = np.array(iKxy)
    # Get the indices to sort Kr from low to high
    iKr_sorted = np.argsort(Kr, axis = 0)
    # Sort Kr, reorder Kx, Ky indices according to sorted Kr
    Kr_sorted, iKxy_sorted = Kr[iKr_sorted], iKxy[iKr_sorted]
    print('\nMerging x and y components into r...')
    # Sum of any RFft that has equal Kr
    # For each Kr_sorted[i]
    RijFft_r, RiiFft_r = [], []
    i = 0
    # Go through every Kr_sorted[i]
    while i <= len(iKr_sorted) - 1:
        # Store current RFft value
        # iKxy_sorted[:, 1] is y
        RijFft_r_i, RiiFft_r_i = RijFft[iKxy_sorted[i, 1], iKxy_sorted[i, 0]], RiiFft[iKxy_sorted[i, 1], iKxy_sorted[i, 0]]
        match = 0
        # Go through every Kr_sorted[j, j > i]
        for j in range(i + 1, len(iKr_sorted)):
            # If Kr_sorted[j] = Kr_sorted[i], add up RFft
            if Kr_sorted[j] == Kr_sorted[i]:
                match += 1
                # Recall Kx/Ky is the same order of RFft's column/row
                RijFft_r_i += RijFft[iKxy_sorted[j, 1], iKxy_sorted[j, 0]]
                RiiFft_r_i += RiiFft[iKxy_sorted[j, 1], iKxy_sorted[j, 0]]
            # Since Kr_sorted is sorted from low to high, if there's no match, then Kr[i] is unique, proceed to Kr[i + 1]
            else:
                break

        # Add summed RFft to list
        RijFft_r.append(RijFft_r_i)
        RiiFft_r.append(RiiFft_r_i)
        # Jump i in case of matches
        i += match + 1

    # Get the unique Kr_sorted
    Kr_sorted = np.unique(Kr_sorted)
    Eij, Eii = np.array(RijFft_r), np.array(RiiFft_r)

    return RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr_sorted, TKE


@timer
# jit() gives unknown error
# @jit(parallel = True, fastmath = True)
def getPlanarEnergySpectrum2(u2D, v2D, w2D, cellSizes, fftNorm = None, mergeXY = True):
    # Velocity fluctuations
    # The mean here is spatial average in slice plane
    # The Taylor hypothesis states that for fully developed turbulence,
    # the spatial average and the time average are equivalent
    uRes2D, vRes2D, wRes2D = u2D - u2D.mean(), v2D - v2D.mean(), w2D - w2D.mean()
    TKE_xy = 0.5*np.mean(uRes2D**2 + vRes2D**2 + wRes2D**2)
    # Number of samples in x and y
    nPtX, nPtY = uRes2D.shape[1], uRes2D.shape[0]
    # Take the upper bound integer of half of nPt
    # For odd points, take (N + 1)/2
    # For even points, take N/2
    nPtKx, nPtKy = int(np.ceil(nPtX/2)), int(np.ceil(nPtY/2))
    uResFft, vResFft, wResFft = np.fft.fft2(uRes2D, axes = (0, 1), norm = fftNorm)[:nPtKy, :nPtKx], \
                                np.fft.fft2(vRes2D, axes = (0, 1), norm = fftNorm)[:nPtKy, :nPtKx], \
                                np.fft.fft2(wRes2D, axes = (0, 1), norm = fftNorm)[:nPtKy, :nPtKx]
    # Corresponding frequency in x and y directions, expressed in cycles/m
    # Kx corresponds to k/n in np.fft.fft documentation, where
    # k = 0, ..., (n - 1)/2; n is number of samples
    # Number of columns are number of x
    # d is sample spacing, which should be equidistant,
    # in this case, cell size in x and y respectively
    # Again, only take the first half, the positive frequencies
    Kx, Ky = np.fft.fftfreq(nPtX, d = cellSizes[0])[:nPtKx], np.fft.fftfreq(nPtY, d = cellSizes[1])[:nPtKy]
    # Kx and Ky is defined as 2Npi/L, while the K in np.fft.fftn() is simply N/pi
    # Thus scale old Kx, Ky by 2pi
    Kx *= 2*np.pi
    Ky *= 2*np.pi
    # Because Kx, Ky were scaled with 2pi,
    # u(Kold) becomes u(Kold*2pi)/2pi
    # => 1/2pi*DFT(u(x)).
    # Then for 2D, it's (1/2pi)^2
    uResFft, vResFft, wResFft = uResFft/(2*np.pi)**2, vResFft/(2*np.pi)**2, wResFft/(2*np.pi)**2
    # Before calculating (cross-)correlation, add arrays to 2 tuples
    U1ResFft = (uResFft, uResFft, uResFft, vResFft, vResFft, wResFft)
    U2ResFft = (uResFft, vResFft, wResFft, vResFft, wResFft, wResFft)
    # 2D DFT of Rij will be half of Rij sizes since the other half are just conjugates that's not interesting, given that Rij(x, y) are all real
    RijFft = np.empty((nPtKy, nPtKx, 6), dtype = np.complex128)
    # The 6 components are 11, 12, 13,
    # 22, 23,
    # 33
    for i in prange(6):
        # Perform the 2-point (cross-)correlation
        RijFft[:, :, i] = np.multiply(U1ResFft[i], np.conj(U2ResFft[i]))

    # Trace of 2-point correlations
    RiiFft = RijFft[:, :, 0] + RijFft[:, :, 3] + RijFft[:, :, 5]
    # RiiFft[0, 0] is the sum of all 2-point Rii by definition
    # # For TKE, it has to be 1-point Rij integral over K
    # TKE = np.abs(0.5*RiiFft[0, 0])

    if mergeXY:
        # Resultant K after combining Kx and Ky
        Kr, iKxy = np.empty(nPtKx*nPtKy), []
        cnt = 0
        # Go through each Kx, then Ky
        for i in prange(len(Kx)):
            for j in range(len(Ky)):
                # Take the resultant
                Kr[cnt] = np.sqrt(Kx[i]**2 + Ky[j]**2)
                # To access i/j, use iKxy[row][col] format since iKxy is a 2D list instead of 2D np.ndarray
                iKxy.append((i, j))
                cnt += 1

        iKxy = np.array(iKxy)
        # Get the indices to sort Kr from low to high
        iKr_sorted = np.argsort(Kr, axis = 0)
        # Sort Kr, reorder Kx, Ky indices according to sorted Kr
        Kr_sorted, iKxy_sorted = Kr[iKr_sorted], iKxy[iKr_sorted]
        print('\nMerging x and y components into r...')
        # Integrate any RFft that has equal Kr, by dS(Kr)
        # For each Kr_sorted[i]
        RijFft_r, RiiFft_r = [], []
        i = 0
        # Go through every Kr_sorted[i]
        while i <= len(iKr_sorted) - 1:
            # Store current RFft value
            # iKxy_sorted[:, 1] is y
            RijFft_r_i, RiiFft_r_i = RijFft[iKxy_sorted[i, 1], iKxy_sorted[i, 0]], RiiFft[iKxy_sorted[i, 1], iKxy_sorted[i, 0]]
            match = 0
            # Go through every Kr_sorted[j, j > i]
            for j in range(i + 1, len(iKr_sorted)):
                # If Kr_sorted[j] = Kr_sorted[i], add up RFft
                if Kr_sorted[j] == Kr_sorted[i]:
                    match += 1
                    # Recall Kx/Ky is the same order of RFft's column/row
                    RijFft_r_i += RijFft[iKxy_sorted[j, 1], iKxy_sorted[j, 0]]
                    RiiFft_r_i += RiiFft[iKxy_sorted[j, 1], iKxy_sorted[j, 0]]
                # Since Kr_sorted is sorted from low to high, if there's no match, then Kr[i] is unique, proceed to Kr[i + 1]
                else:
                    break

            # Assuming RijFft_r(Kr[i]) on Kr[i] are equidistant,
            # Then integral of RijFft_r at Kr[i] ring is
            # dS(Kr[i])*sum(RijFft_r[Kr[i]]),
            # where dS(Kr[i]) = 2piKr[i]/N_RijFft_r[Kr[i]]
            RijFft_r_i *= 2*np.pi*Kr[i]/(match + 1)
            RiiFft_r_i *= 2*np.pi*Kr[i]/(match + 1)
            # Add integrated RFft to list
            RijFft_r.append(RijFft_r_i)
            RiiFft_r.append(RiiFft_r_i)
            # Jump i in case of matches
            i += match + 1

        # Get the unique Kr_sorted
        Kr_sorted = np.unique(Kr_sorted)
        # Eij, Eii are scaled by 0.5 due to TKE = 0.5*ui^2 = integral(Eii(Kr)) over dKr
        Eij, Eii = 0.5*np.array(RijFft_r), 0.5*np.array(RiiFft_r)
    # If decompose the energy spectrum into x and y
    else:
        Kr_sorted = np.empty_like(Kx)
        # Decompose the DFT 2D arrays
        Eij_x, Eij_y = np.empty((nPtKx, 6), dtype = np.complex128), np.empty((nPtKy, 6), dtype = np.complex128)
        Eii_x, Eii_y = np.empty(nPtKx, dtype = np.complex128), np.empty(nPtKy, dtype = np.complex128)
        # For x components, compute the average over y, for each Kx so that only one E exists for each K
        # Then integrate Rijfft, RiiFft by circumference of equal K since RijFft ~ m^4/s^2 and target Eij ~ m^3/s^2
        # dKx, dKy = Kx[1] - Kx[0], Ky[1] - Ky[0]
        # x is column, y is row
        for i in prange(nPtKx):
            for j in range(6):
                Eij_x[i, j] = 0.5*np.mean(RijFft[:, i, j])*2*np.pi*Kx[i]

            Eii_x[i] =  0.5*np.mean(RiiFft[:, i])*2*np.pi*Kx[i]

        for i in prange(nPtKy):
            for j in range(6):
                Eij_y[i, j] = 0.5*np.mean(RijFft[i, :, j])*2*np.pi*Ky[i]

            Eii_y[i] = 0.5*np.mean(RiiFft[i, :])*2*np.pi*Ky[i]

        Eij, Eii = (np.abs(Eij_x), np.abs(Eij_y)), (np.abs(Eii_x), np.abs(Eii_y))

    # TKE = 0.5*ui^2 = integral(Eii(Kr)) over dKr
    # Do this by summing up the area between each dKr
    TKE = 0.
    # for i in prange(len(Kr_sorted) - 1):
    #     TKE += (Eii[i] + Eii[i + 1])/2.*(Kr_sorted[i + 1] - Kr_sorted[i])

    return RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr_sorted, TKE


t0 = time.time()
# Read 2D mesh and velocity
x2D, y2D, z2D, U2D, u2D, v2D, w2D = PostProcess_EnergySpectrum.readStructuredSliceData(sliceName = sliceName, case = case, caseDir = caseDir, time = sliceTime)

# [DEPRECATED]
# E, Evert, Kr = PostProcess_EnergySpectrum.getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = np.array([10., 10.]))
t1 = time.time()
#print(f'\nFinished readSliceRawData in {t1 - t0} s')
print('\nFinished readSliceRawData in {:.4f} s'.format(t1 - t0))

# Calculate 2-point (cross-)correlation and energy spectrum density and corresponding wave number
# [DEPRECATED]
# uResFft, vResFft, wResFft, freqX, freqY = getSliceEnergySpectrum(u2D, v2D, w2D, cellSizes = [10., 10.])
RiiFft, Eii, RijFft, Eij, Kx, Ky, Kr, TKE = getPlanarEnergySpectrum2(u2D, v2D, w2D, cellSizes = cellSizes, fftNorm = fftNorm, mergeXY = mergeXY)

# If Kx and Ky were not merged, repeat the following process twice, 1st x then y
loop = 1 if mergeXY else 2
# Create a copy of Eij and Eii so that the copies are not touched below
Eij_copy, Eii_copy = Eij, Eii
xLabels = ('$K_r$ [1/m]',) if mergeXY else ('$K_x$ [1/m]', '$K_y$ [1/m]')
for l in prange(loop):
    if not mergeXY:
        Kr = Kx if l == 0 else Ky
        Eij, Eii = Eij_copy[l], Eii_copy[l]

    Kr *= krFactor
    # Convert complex value to real and normalize it if enabled
    # norm = len(Eii) if normalize else 1
    # norm = u2D.shape[0]*u2D.shape[1] if normalize else 1
    norm = Kr*Lt if plotNorm else 1
    # Convert complex number to real
    Eii, Eij = np.abs(Eii), np.abs(Eij)
    # Eii, Eij, TKE = np.abs(Eii)/norm, np.abs(Eij)/norm, TKE/norm
    # Normalize Eij and Eii sequentially, thus 7 iterations
    # Order: 11, 12, 13,
    # 22, 23,
    # 33,
    # ii
    if plotNorm:
        for i in range(7):
            if i == 6:
                Eii = np.divide(Eii, norm)
            else:
                Eij[:, i] = np.divide(Eij[:, i], norm)


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
    Binning Operation
    """
    # Kr manipulation
    # Kr *= krFactor
    # Mean cell size, used for the fill-in plot of the SFS region
    cellSize = np.mean(cellSizes)
    # Target Kr after binning
    Kr_binned = np.logspace(-3, 0, nBin)
    # Kr_binned = np.linspace(Kr.min(), Kr.max(), 1000)
    # Indices of which bin each Kr would fall in
    iBin = np.digitize(Kr, Kr_binned)
    # Get all unique iBin values
    iBin_uniq = np.unique(iBin)
    # Go through every index of targeted bins
    for i in range(nBin):
        # If such index not found in iBin, it means the bins in that region is too refined for binning old values
        # E.g. Kr = [1, 2, 3],
        # Kr_binned = [1, 1.5, 2, 2.5, 3],
        # no Kr to put in Kr_binned of 1.5 or 2.5
        # Make those over-flown Kr bin NaN
        if i not in iBin_uniq:
            Kr_binned[i] = np.nan

    # Then remove those NaN entries
    Kr_binned = Kr_binned[~np.isnan(Kr_binned)]
    # Refresh the bin indices that each Kr should fall in
    iBin = np.digitize(Kr, Kr_binned)
    # Then, for all E that fall in a bin, get the average
    # Go through every Eij column, 6 in total, lastly, do it for Eii too
    Eij_binned_0 = np.array([np.mean(Eij[iBin == i, 0]) for i in range(len(Kr_binned))])
    Eij_binned = np.empty((Eij_binned_0.shape[0], 6))
    Eij_binned[:, 0] = Eij_binned_0
    for j in range(1, 7):
        # For Eij
        if j != 6:
            Eij_binned[:, j] = np.array([np.mean(Eij[iBin == i, j]) for i in range(len(Kr_binned))])
        # For Eii
        else:
            Eii_binned = np.array([np.mean(Eii[iBin == i]) for i in range(len(Kr_binned))])


    """
    Plot the Energy Spectrums of Eii, E12, E33
    """
    # E to be plotted into a tuple
    E = (Eii_binned, Eij_binned[:, 1], Eij_binned[:, -1])
    # Figure order is Eii, E12, E33
    if mergeXY:
        figNames = (case + '_Eii', case + '_E12', case + '_E33')
    else:
        figNames = (case + '_Eii_' + str(l), case + '_E12_' + str(l), case + '_E33_' + str(l))

    if mergeXY:
        yLabels = (r'$E_{ii}(K_r)$ [m$^3$/s$^2$]', r'$E_{12}(K_r)$ [m$^3$/s$^2$]', r'$E_{33}(K_r)$ [m$^3$/s$^2$]')
    else:
        if l == 0:
            yLabels = (r'$E_{ii}(K_x)$ [m$^3$/s$^2$]', r'$E_{12}(K_x)$ [m$^3$/s$^2$]', r'$E_{33}(K_x)$ [m$^3$/s$^2$]')
        else:
            yLabels = (r'$E_{ii}(K_y)$ [m$^3$/s$^2$]', r'$E_{12}(K_y)$ [m$^3$/s$^2$]', r'$E_{33}(K_y)$ [m$^3$/s$^2$]')
    # Go through Eii, E12, E33 and plot each of them
    for i in range(3):
        # If plotting E12
        if i == 1:
            xList, yList = (Kr_binned, E12ref[:, 0], Kr), (E[i], E12ref[:, 1], E_Kolmo)
            plotsLabel = (label, 'Churchfield et al.', 'Kolmogorov model')
        # Else if plotting E33
        elif i == 2:
            xList, yList = (Kr_binned, E33ref[:, 0], Kr), (E[i], E33ref[:, 1], E_Kolmo)
            plotsLabel = (label, 'Churchfield et al.', 'Kolmogorov model')
        # Else if plotting Eii, no reference data available
        else:
            xList, yList = (Kr_binned, Kr), (E[i], E_Kolmo)
            plotsLabel = (label, 'Kolmogorov model')

        # Initialize figure
        plot = Plot2D(xList, yList, xLabel = xLabels[l], yLabel = yLabels[i], name = figNames[i], save = save, show = show, xLim = xLim, yLim = yLim, figDir = resultPath)
        plot.initializeFigure()
        plot.plotFigure(plotsLabel = plotsLabel)
        plot.axes[0].fill_between((1/cellSize, xLim[1]), yLim[0], yLim[1], alpha = fillAlpha, facecolor =
        plot.gray, zorder = -1)
        plot.finalizeFigure(xyScale = ('log', 'log'))




# [DEPRECATED]
@timer
@jit(parallel = True, fastmath = True)
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
    for iX in prange(len(freqX)):
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
        for j in prange(i + 1, len(KrSorted)):
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
