
import numpy as np
from mayavi.mlab import *
from mayavi.modules.contour_grid_plane import ContourGridPlane

def mytest_contour3d(trans):
    x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]

    scalars = x * x * 0.5 + y * y + z * z * 2.0

    obj = contour3d(scalars, contours=4, transparent=trans, opacity = 1)
    return obj


from mayavi.api import Engine
engine = Engine()
engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
# --------------------------
engine.new_scene()
a = mytest_contour3d(False)
cgp = ContourGridPlane()
engine.add_module(cgp)
cgp.grid_plane.axis = 'y'
show()
