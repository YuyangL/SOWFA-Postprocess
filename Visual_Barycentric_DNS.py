import numpy as np
from PostProcess_AnisotropyTensor import processAnisotropyTensor_Uninterpolated
import time as t
from mayavi import mlab
from mayavi.api import Engine
import pickle
import os

caseDir, caseName = 'J:/DNS', 'ConvergingDivergingChannel'

dataDir = caseDir + '/' + caseName + '/'
meshName, propertyName = 'meshDNS', 'Y_DNS'
resultFolder = 'Result'

resultDir = dataDir + resultFolder + '/'
try:
    os.makedirs(resultDir)
except OSError:
    pass

# Using encoding = 'latin1' to avoid UnicodeDecoderError, seem to be a bug of Python 3
mesh = pickle.load(open(dataDir + meshName + '.p', 'rb'), encoding = 'latin1')
property = pickle.load(open(dataDir + propertyName + '.p', 'rb'), encoding = 'latin1')

# mesh was in dimension 3 x ..., swap it to last so that ... x 3
mesh = np.swapaxes(mesh, 0, 1)
mesh = np.swapaxes(mesh, 1, 2)
# property was in dimension 9 x ..., swap it to last so that ... x 9
property = np.swapaxes(property, 0, 1)
ccx, ccy, ccz = mesh[:, :, 0].ravel(), mesh[:, :, 1].ravel(), mesh[:, :, 2].ravel()

# Process anisotropy tensor
t0 = t.time()
property, tensors, eigVals3D, eigVecs4D = processAnisotropyTensor_Uninterpolated(property, makeAnisotropic = False)
t1 = t.time()
ticToc = t1 - t0

# Eigenvector manipulation, flip it by -1
eigVecs4D *= -1.


"""
Mayavi Quiver Plot
"""
engine = Engine()
engine.start()
# if len(engine.scenes) == 0:
#     engine.new_scene()

mlab.figure(caseName + '_quivers', engine = engine, size = (1000, 800), bgcolor = (1, 1, 1), fgcolor = (0.5, 0.5, 0.5))
quiver = mlab.quiver3d(ccx, ccy, ccz, eigVecs4D[:, :, 0, 0].ravel(), eigVecs4D[:, :, 0, 1].ravel(), eigVecs4D[:, :, 0, 2].ravel(), scalars = eigVals3D[:, :, 0].ravel(), colormap = 'plasma', mask_points = 5, scale_mode = 'scalar')
mlab.outline()
quiver.glyph.color_mode = 'color_by_scalar'
mlab.savefig(resultDir + caseName + '_quiver.png')
mlab.show()



