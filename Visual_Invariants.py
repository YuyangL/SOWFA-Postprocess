import numpy as np
from PlottingTool import PlotSurfaceSlices3D

"""
Plot Settings
"""
figWidth = 'half'
figDir, name = 'R:', 'thirdInvariant'
show, save = True, True
equalAxis, cbarOrientate = False, 'vertical'
tightLayout, pad, fraction = False, 0.05, 0.1
xLabel, yLabel, zLabel, cmapLabel = r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$III^{\textbf{b}}$'
viewAngle, showCbar = (15, -10), True

"""
User Inputs
"""
nPt = 100


lam1, lam2 = np.linspace(-1/3., 1/3., nPt), np.linspace(-1/3., 1/3., nPt)

lam1, lam2 = np.meshgrid(lam1, lam2)

# Since sum of eigen values in rank-2 anistropy tensor should 0
lam3 = 0 - lam1 - lam2

thirdInv = -3*lam1*lam2*(lam1 + lam2)

thirdInvPlot = PlotSurfaceSlices3D(lam1, lam2, lam3, thirdInv, show = show, save = save, figDir = figDir, figWidth = figWidth, equalAxis = equalAxis, cbarOrientate = cbarOrientate, xLabel = xLabel, yLabel = yLabel, zLabel = '', cmapLabel = cmapLabel, name = name, viewAngles = viewAngle)
thirdInvPlot.initializeFigure()
thirdInvPlot.plotFigure()
thirdInvPlot.finalizeFigure(tightLayout = tightLayout, pad = pad, fraction = fraction, showCbar = showCbar)



