import numpy as np
from PlottingTool import Plot2D, Plot2D_InsetZoom, PlotSurfaceSlices3D, PlotContourSlices3D
try:
    import PostProcess_AnisotropyTensor as PPAT
except ImportError:
    raise ImportError('\nNo module named PostProcess_AnisotropyTensor. Check setup.py and run'
                      '\npython setup.py build_ext --inplace')

try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
from matplotlib.path import Path
import PostProcess_SliceData as PPSD

"""
User Inputs
"""
time = 'latest'
caseDir = '/media/yluan'
caseName = 'ALM_N_H_ParTurb'
# Property names
epsilonSGSmeanName, nuSGSmeanName = 'epsilonSGSmean', 'nuSGSmean'
kResolvedName, kSGSmeanName = 'kResolved', 'kSGSmean'
gradUAvgName, uuPrime2Name = 'gradUAvg', 'uuPrime2'
# sliceNames = ('rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'sevenDdownstreamTurbine')
sliceNames = ('hubHeight', 'quarterDaboveHub', 'turbineApex')
# Subscript for the slice names
sliceNamesSub = 'Slice'
# Height of the horizontal slices, only used for 3D horizontal slices plot
horSliceOffsets = (1, 2, 3, 4, 5)
resultFolder = 'Result'
# Confine the region of interest, list grows with number of slices
# (800, 1800, 800, 1800, 0, 405) good for rotor plane slice
# (600, 2000, 600, 2000, 0, 405) good for hubHeight slice
# (800, 2200, 800, 2200, 0, 405) good for along wind slice
# Overriden in Process User Inputs
confineBox = ((800, 1800, 800, 1800, 0, 405),)  # (None,), ((xmin, xmax, ymin, ymax, zmin, zmax),)
# Orientation of x-axis in x-y plane, in case of angled flow direction
# Only used for values decomposition and confineBox
# Angle in rad and counter-clockwise
xOrientate = 6/np.pi
# Turbine radius, only used for confineBox
r = 63
# For calculating total <epsilon> only
nu = 1e-5


"""
Data Processing Settings
"""
# Magnitude cap for Sij, Rij and scalar basis when calculating them
capSijRij, capSB = 1e9, 1e9
# Whether standardize 5 scalar bases and/or 10 tensor bases
scaleSB, scaleTB = True, True


"""
Plot Settings
"""
# Which type(s) of plot to make
plotType = '3D'  # '2D', '3D', 'all'
# Total number cells intended to plot via interpolation
targetCells = 1e5
interpMethod = 'cubic'
# Number of contours, only for 2D plots or 3D horizontal slice plots
contourLvl = 100
# Label of the property, could be overridden below
valLabel = 'Data'
ext = 'png'
show, save = False, True
dpi = 600


"""
Process User Inputs
"""
# Create tuple to read in batch
propertyNames = (epsilonSGSmeanName, nuSGSmeanName, kResolvedName, kSGSmeanName, gradUAvgName)
# Ensure sliceNames is a tuple
sliceNames = (sliceNames,) if isinstance(sliceNames, str) else sliceNames
# Confined region auto definition
if caseName == 'ALM_N_H_OneTurb':
    # For rotor plane vertical slices
    if sliceNames[0] == 'rotorPlane':
        confineBox = ((1023.583 + r/2, 1212.583 - r/2, 1115.821 + r/2,
                       1443.179 - r/2,
                       0,
                       216 - r/2),
                      (1132.702, 1321.702, 1178.821, 1506.179, 0, 216),
                      (1350.94, 1539.94, 1304.821, 1632.179, 0, 216),
                      (1569.179, 1758.179, 1430.821, 1758.179, 0, 216))

    # For horizontal slices
    elif sliceNames[0] in ('groundHeight', 'hubHeight'):
        # Confinement for z doesn't matter since the slices are horizontal
        confineBox = ((600, 2000, 600, 2000, 0, 216),)*len(sliceNames)

elif caseName == 'ALM_N_H_ParTurb':
    if sliceNames[0] == 'rotorPlane':
        confineBox = ((1023.583 + r/2, 1212.583 - r/2, 1115.821 + r/2,
                       1443.179 - r/2,
                       0,
                       216 - r/2),
                      (1132.702, 1321.702, 1178.821, 1506.179, 0, 216),
                      (1350.94, 1539.94, 1304.821, 1632.179, 0, 216),
                      (1569.179, 1758.179, 1430.821, 1758.179, 0, 216))
    elif sliceNames[0] in ('groundHeight', 'hubHeight'):
        # Confinement for z doesn't matter since the slices are horizontal
        confineBox = ((700, 2300, 700, 2300, 0, 216),)*len(sliceNames)

# Automatic viewAngle and figure settings, only for 3D plots
if sliceNames[0] == 'rotorPlane':
    viewAngle = (20, -100)
    equalAxis, figSize, figWidth = True, (1, 1), 'full'
elif sliceNames[0] in ('groundHeight', 'hubHeight'):
    viewAngle = (15, -115)
    equalAxis, figSize, figWidth = False, (2, 1), 'half'
else:
    viewAngle = (20, -100)
    equalAxis, figSize, figWidth = True, (1, 1), 'full'

# Unify plotType user inputs
if plotType in ('2D', '2d'):
    plotType = '2D'
elif plotType in ('3D', '3d'):
    plotType = '3D'
elif plotType in ('all', 'All', '*'):
    plotType = 'all'

if 'kResolved' in propertyNames and 'kSGSmean' in propertyNames:
    valLabel = r'$\langle k\rangle$ [m$^2$/s$^2$]'
elif 'epsilonSGSmean' in propertyNames and 'nuSGSmean' in propertyNames:
    valLabel = r'$\langle \epsilon \rangle$ [m$^2$/s$^3$]'
elif 'gradUAvg' in propertyNames:
    valLabel = (r'$\nabla \textrm{U}$ [1/s]')


"""
Read Slice Data
"""
# Initialize case
case = PPSD.SliceProperties(time = time, caseDir = caseDir, caseName = caseName, xOrientate = xOrientate, resultFolder =
resultFolder)
# Read slices
case.readSlices(propertyNames = propertyNames, sliceNames = sliceNames, sliceNamesSub = sliceNamesSub)


"""
Process TBNN Inputs and Plot
"""
# Go through specified slices and plot 5 scalar bases
for i in range(len(sliceNames)):
    epsilonSGSmean = case.slicesVal[epsilonSGSmeanName + '_' + sliceNames[i]]
    nuSGSmean = case.slicesVal[nuSGSmeanName + '_' + sliceNames[i]]
    kResolved = case.slicesVal[kResolvedName + '_' + sliceNames[i]]
    kSGSmean = case.slicesVal[kSGSmeanName + '_' + sliceNames[i]]
    gradUAvg = case.slicesVal[gradUAvgName + '_' + sliceNames[i]]
    # uuPrime2 = case.slicesVal[uuPrime2Name + '_' + sliceNames[i]]
    # Calculate total <TKE>
    print(' Calculating total <TKE> for {}...'.format(sliceNames[i]))
    kMean = kResolved + kSGSmean
    # Calculate total turbulent dissipation rate <epsilon>
    print(' Calculating total <epsilon> for {}...'.format(sliceNames[i]))
    epsilonMean = case.calcSliceMeanDissipationRate(epsilonSGSmean, nuSGSmean, nu)
    # Since gradUAvg is a vector field and uuPrime2 is a double spatial correlation tensor field
    # transform them to a coordinate system aligned with wind direction at hub height
    # gradUAvg, uuPrime2 = case.rotateSpatialCorrelationTensors((gradUAvg, uuPrime2), rotateXY = xOrientate, dependencies = 'x')
    (gradUAvg,) = case.rotateSpatialCorrelationTensors((gradUAvg,), rotateXY = xOrientate,
                                                              dependencies = 'x')
    # Calculate TBNN inputs, incl. 5 scalar bases as input x and 10 tensor bases as final layer transformation
    # Tensor bases should transform scalar bases to the coordinate system aligned to wind direction at hub height
    scalarBasis, tensorBasis, mean, std = case.processSliceTBNN_Inputs(gradUAvg, kMean, epsilonMean, capSijRij = capSijRij, capSB = capSB, scaleSB = scaleSB, scaleTB = scaleTB)
    # Plot labels
    xLabel, yLabel, zLabel = (r'$r$ [m]', r'$z$ [m]', 'Component') \
        if case.slicesOrientate[gradUAvgName + '_' + sliceNames[i]] == 'vertical' else \
        (r'$x$ [m]', r'$y$ [m]', 'Component')
    # Go through property of interest
    listX2D, listY2D, listZ2D, listVals3D = [], [], [], []
    inputNames = ('scalarBasis',)
    for j, vals2D in enumerate((scalarBasis,)):
        # Interpolation so that property is nX x nY/nZ x nComponent
        # Since all slices have the same x, y, z, use the gradUAvg slice's x, y, z
        x2D, y2D, z2D, vals3D = case.interpolateDecomposedSliceData_Fast(case.slicesCoor[gradUAvgName + '_' + sliceNames[i]][:, 0],
                                                                         case.slicesCoor[gradUAvgName + '_' + sliceNames[i]][:, 1],
                                                                         case.slicesCoor[gradUAvgName + '_' + sliceNames[i]][:, 2],
                                                                         vals2D, sliceOrientate = case.slicesOrientate[gradUAvgName + '_' + sliceNames[i]], xOrientate = case.xOrientate, targetCells = targetCells, interpMethod = interpMethod, confineBox = confineBox[i])

        # # Flatten if vals3D only have one component like a scalar field
        # if vals3D.shape[2] == 1:
        #     vals3D = vals3D.reshape((vals3D.shape[0], vals3D.shape[1]))

        # If vals3D has more than one component, append the components to list for plotting
        if vals3D.shape[2] > 1:
            for iComp in range(vals3D.shape[-1]):
                listX2D.append(x2D), listY2D.append(y2D), listZ2D.append(z2D), listVals3D.append(vals3D[:, :, iComp])


        # Determine the unit along the vertical slice since it's angled and
        if case.slicesOrientate[gradUAvgName + '_' + sliceNames[i]] == 'vertical':
            if confineBox[0] is None:
                lx = np.max(x2D) - np.min(x2D)
                ly = np.max(y2D) - np.min(y2D)
            else:
                lx = confineBox[i][1] - confineBox[i][0]
                ly = confineBox[i][3] - confineBox[i][2]

            r2D = np.linspace(0, np.sqrt(lx**2 + ly**2), x2D.shape[0])


        """
        Plot 5 Scalar Bases in 1 figure for a slice location
        """
        if case.slicesOrientate[gradUAvgName + '_' + sliceNames[i]] == 'horizontal':
            slicePlot3D = PlotContourSlices3D(listX2D, listY2D, listVals3D, horSliceOffsets, contourLvl = contourLvl,
                                              gradientBg = False, name = inputNames[j] + '_' + sliceNames[i], xLabel = xLabel, yLabel = yLabel,
                                              zLabel = zLabel, cmapLabel = valLabel, save = save, show = show,
                                              figDir = case.resultPath, viewAngles = viewAngle, figWidth = figWidth,
                                              equalAxis = equalAxis, cbarOrientate = 'vertical')
        elif case.slicesOrientate[gradUAvgName + '_' + sliceNames[i]] == 'vertical':
            slicePlot3D = PlotSurfaceSlices3D(listX2D, listY2D, listZ2D, listVals3D, xLabel = xLabel, yLabel = yLabel,
                                              zLabel = zLabel, cmapLabel = valLabel, name = str(sliceNames),
                                              save = save, show = show, figDir = case.resultPath,
                                              viewAngles = viewAngle, figWidth = figWidth, equalAxis = equalAxis,
                                              cbarOrientate = 'horizontal')

        slicePlot3D.initializeFigure(figSize = figSize)
        slicePlot3D.plotFigure()
        slicePlot3D.finalizeFigure()







