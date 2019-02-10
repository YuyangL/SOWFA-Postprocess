import numpy as np
from PlottingTool import PlotSurfaceSlices3D

"""
Plot Settings
"""
figWidth = 'half'
figDir, name = 'R:', 'thirdInvariant'
show, save = False, True
equalAxis, cbarOrientate = False, 'vertical'
tightLayout, pad, fraction = False, 0.05, 0.1
xLabel, yLabel, zLabel, cmapLabel = r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\mathit{III}^{b}$'
# View angles best be (15, -10) and (15, -100)
viewAngle, showCbar = (15, -10), True
xLim = yLim = zLim = (-1/3., 2/3.)

"""
User Inputs
"""
nPt = 250


lam1, lam2 = np.linspace(-1/3., 2/3., nPt), np.linspace(-1/3., 2/3., nPt)

lam1, lam2 = np.meshgrid(lam1, lam2)
sumlam12 = lam1 + lam2
# Max of lambda is 2/3. Min of lambda is -1/3.
lam1[sumlam12 > 1/3.], lam2[sumlam12 > 1/3.] = np.nan, np.nan

# Since sum of eigen values in rank-2 anistropy tensor should 0
lam3 = 0 - lam1 - lam2

thirdInv = -3*lam1*lam2*(lam1 + lam2)

thirdInvPlot = PlotSurfaceSlices3D(lam1, lam2, lam3, thirdInv, show = show, save = save, figDir = figDir, figWidth = figWidth, equalAxis = equalAxis, cbarOrientate = cbarOrientate, xLabel = xLabel, yLabel = yLabel, zLabel = zLabel, cmapLabel = cmapLabel, name = name, viewAngles = viewAngle, xLim = xLim, yLim = yLim, zLim = zLim)
thirdInvPlot.initializeFigure()
thirdInvPlot.plotFigure()
thirdInvPlot.finalizeFigure(pad = pad, fraction = fraction, showCbar = showCbar)



