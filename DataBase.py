import numpy as np

def OneTurb(slice_orient='vertical', r=63., pitch=5., zhub=90.):
    # Northern border x, y; southern border x, y
    if 'hor' in slice_orient:
        turb_borders = [[1086.583, 1334.06, 1149.583, 1224.94]]
    # Lower border x, z; upper border x, z
    else:
        turb_borders = [[1086.583 - r*np.sin(pitch/180.*np.pi), zhub - r, 1149.583 + r*np.sin(pitch), zhub + r]]

    # Turbine center x, y
    turb_centers = [[1118.083, 1279.5]]
    # Turbine centers x, y, z at 1D upstream, rotor plane, 1D downstream
    # then 3D downstream, 5D downstream, 7D downstream
    turb_centers_frontview = ((1008.964, 1216.5, 90.),
                              (1118.083, 1279.5, 90.),
                              (1227.202, 1342.5, 90.),
                              (1445.44, 1468.5, 90.),
                              (1663.679, 1594.5, 90.),
                              (1881.917, 1720.5, 90.))
    # Confine box for plotting only the interesting region
    if 'vert' in slice_orient:
        # 1D upstream, rotor plane, 1D downstream: xmin, xmax, ymin, ymax, zmin, zmax
        confinebox = ((914.464, 1103.464, 1052.821, 1380.179, 0., 216.),
                      (1023.583, 1212.583, 1115.821, 1443.179, 0., 216.),
                      (1132.702, 1321.702, 1178.821, 1506.179, 0., 216.),
                      (1350.94, 1539.94, 1304.821, 1632.179, 0., 216.),
                      (1569.179, 1758.179, 1430.821, 1758.179, 0., 216.),
                      (1787.417, 1976.417, 1556.821, 1884.179, 0., 216.))

        # 3D, 5D, 7D downstream: xmin, xmax, ymin, ymax, zmin, zmax
        confinebox2 = ((1350.94, 1539.94, 1304.821, 1632.179, 0., 216.),
                       (1569.179, 1758.179, 1430.821, 1758.179, 0., 216.),
                       (1787.417, 1976.417, 1556.821, 1884.179, 0., 216.))
    else:
        confinebox = ((800, 2400, 800, 2400, 0, 216),)*10
        confinebox2 = confinebox

    return turb_borders, turb_centers_frontview, confinebox, confinebox2
