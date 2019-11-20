import numpy as np

def OneTurb(slice_orient='vertical', r=63., pitch=5., zhub=90., rot_z=np.pi/6., rotate_field=True):
    # Northern border x, y; southern border x, y
    if 'hor' in slice_orient:
        if not rotate_field:
            turb_borders = [[1086.583, 1334.06, 1149.583, 1224.94]]
        else:
            turb_borders = [[1086.583*np.cos(rot_z) + np.sin(rot_z)*1334.06, 1334.06*np.cos(rot_z) - 1086.583*np.sin(rot_z),
                             1149.583*np.cos(rot_z) + np.sin(rot_z)*1224.94, 1224.94*np.cos(rot_z) - 1149.583*np.sin(rot_z)]]

    # Lower border x, z; upper border x, z
    else:
        turb_borders = [[1118.083 - r*np.sin(pitch/180.*np.pi), zhub - r, 1118.083 + r*np.sin(pitch), zhub + r]]

    # Turbine centers x, y, z at 1D upstream, rotor plane, 1D downstream
    # then 3D downstream, 5D downstream, 7D downstream
    turb_centers_frontview = ((1008.964, 1216.5, zhub),
                              (1118.083, 1279.5, zhub),
                              (1227.202, 1342.5, zhub),
                              (1445.44, 1468.5, zhub),
                              (1663.679, 1594.5, zhub),
                              (1881.917, 1720.5, zhub))
    # Confine box for plotting only the interesting region
    if 'vert' in slice_orient:
        # 1D upstream, rotor plane, 1D downstream, 3D downstream, 5D downstream, 7D downstream:
        # xmin, xmax, ymin, ymax, zmin, zmax
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
        if not rotate_field:
            confinebox = ((1000, 1700., 1100, 1800., 0., 216.),)*10
        else:
            confinebox = ((994.344*np.cos(rot_z) + 989.821*np.sin(rot_z), 994.344*np.cos(rot_z) + 989.821*np.sin(rot_z) + 882.,
                           989.821*np.cos(rot_z) - 994.344*np.sin(rot_z) - 252, 989.821*np.cos(rot_z) - 994.344*np.sin(rot_z) + 882 - 252, 0., 216.),)*10

        confinebox2 = confinebox

    return turb_borders, turb_centers_frontview, confinebox, confinebox2


def ParTurb(slice_orient='vertical', r=63., pitch=5., zhub=90., rot_z=np.pi/6., rotate_field=True):
    # Northern border x, y, southern border x, y of southern turbine
    # then northern border x, y, southern border x, y of northern turbine
    if 'hor' in slice_orient:
        if not rotate_field:
            turb_borders = [[1212.583, 1115.821, 1275.583, 1006.702],
                            [960.583, 1552.298, 1023.583, 1443.179]]
        else:
            turb_borders = [[1212.583*np.cos(rot_z) + np.sin(rot_z)*1115.821, 1115.821*np.cos(rot_z) - 1212.583*np.sin(rot_z),
                             1275.583*np.cos(rot_z) + np.sin(rot_z)*1006.702, 1006.702*np.cos(rot_z) - 1275.583*np.sin(rot_z)],
                            [960.583*np.cos(rot_z) + np.sin(rot_z)*1552.298, 1552.298*np.cos(rot_z) - 960.583*np.sin(rot_z),
                             1023.583*np.cos(rot_z) + 1443.179*np.sin(rot_z), 1443.179*np.cos(rot_z) - 1023.583*np.sin(rot_z)]]

    # Southern turbine then northern turbine: lower border x, z; upper border x, z
    else:
        turb_borders = [[1244.083 - r*np.sin(pitch/180.*np.pi), zhub - r, 1244.083 + r*np.sin(pitch), zhub + r],
                        [[992.083 - r*np.sin(pitch/180.*np.pi), zhub - r, 992.083 + r*np.sin(pitch), zhub + r]]]

    # Turbine centers x, y, z at 1D upstream, rotor plane, 1D downstream
    # then 3D downstream, 5D downstream, 7D downstream
    # First southern turbine, then northern turbine
    turb_centers_frontview = (
                              (1134.964, 998.262, zhub),  # S
                              (882.964, 1434.738, zhub),  # N
                              (1244.083, 1061.262, zhub),  # S
                              (992.083, 1497.738, zhub),  # N
                              (1353.202, 1124.262, zhub),  # S
                              (1101.202, 1560.738, zhub),  # N
                              (1571.44, 1250.262, zhub),  # S
                              (1319.44, 1686.738, zhub),  # N
                              (1789.679, 1376.262, zhub),  # S
                              (1537.679, 1812.738, zhub),  # N
                              (2007.917, 1502.262, zhub),  # S
                              (1755.917, 1938.738, zhub)  # N
                              # # Northern turbine
                              # (882.964, 1434.738, zhub),
                              # (992.083, 1497.738, zhub),
                              # (1101.202, 1560.738, zhub),
                              # (1319.44, 1686.738, zhub),
                              # (1537.679, 1812.738, zhub),
                              # (1755.917, 1938.738, zhub)
                              )
    # Confine box for plotting only the interesting region
    if 'vert' in slice_orient:
        # 1D upstream, rotor plane, 1D downstream, 3D downstream, 5D downstream, 7D downstream:
        # xmin, xmax, ymin, ymax, zmin, zmax
        confinebox = ((788.464, 1229.464, 834.583, 1598.417, 0., 216.),
                      (897.583, 1338.583, 897.583, 1661.417, 0., 216.),
                      (1006.702, 1447.702, 960.583, 1724.417, 0., 216.),
                      (1224.94, 1665.94, 1086.583, 1850.417, 0., 216.),
                      (1443.179, 1884.179, 1212.583, 1976.417, 0., 216.),
                      (1661.417, 2102.417, 1338.583, 2102.417, 0., 216.))

        # 3D, 5D, 7D downstream: xmin, xmax, ymin, ymax, zmin, zmax
        confinebox2 = ((1224.94, 1665.94, 1086.583, 1850.417, 0., 216.),
                       (1443.179, 1884.179, 1212.583, 1976.417, 0., 216.),
                       (1661.417, 2102.417, 1338.583, 2102.417, 0., 216.))
    else:
        if rotate_field:
            confinebox = ((1120.344*np.cos(rot_z) + 771.583*np.sin(rot_z),
                           882. + 1120.344*np.cos(rot_z) + 771.583*np.sin(rot_z),
                           -1120.344*np.sin(rot_z) + 771.583*np.cos(rot_z),
                           882 - 1120.344*np.sin(rot_z) + 771.583*np.cos(rot_z), 0., 216.),)*10
        else:
            confinebox = ((800., 2400., 800., 2400., 0., 216.),
                          (800., 2400., 800., 2400., 0., 216.))

        confinebox2 = confinebox

    return turb_borders, turb_centers_frontview, confinebox, confinebox2


def SeqTurb(slice_orient='vertical', r=63., pitch=5., zhub=90., rot_z=np.pi/6., rotate_field=True):
    # Northern border x, y, southern border x, y of upwind turbine
    # then northern border x, y, southern border x, y of downwind turbine
    if 'hor' in slice_orient:
        if not rotate_field:
            turb_borders = [[1086.583, 1334.06, 1149.583, 1224.94],
                            [1850.417, 1775.06, 1913.417, 1665.94]]
        else:
            turb_borders = [[1086.583*np.cos(rot_z) + 1334.06*np.sin(rot_z), 1334.06*np.cos(rot_z) - 1086.583*np.sin(rot_z),
                             1149.583*np.cos(rot_z) + 1224.94*np.sin(rot_z), 1224.94*np.cos(rot_z) - 1149.583*np.sin(rot_z)],
                            [1850.417*np.cos(rot_z) + 1775.06*np.sin(rot_z), 1775.06*np.cos(rot_z) - 1850.417*np.sin(rot_z),
                             1913.417*np.cos(rot_z) + 1665.94*np.sin(rot_z), 1665.94*np.cos(rot_z) - 1913.417*np.sin(rot_z)]]

    # Upwind turbine then downwind turbine: lower border x, z; upper border x, z
    else:
        turb_borders = [[1118.083 - r*np.sin(pitch/180.*np.pi), zhub - r, 1118.083 + r*np.sin(pitch), zhub + r],
                        [[1881.917 - r*np.sin(pitch/180.*np.pi), zhub - r, 1881.917 + r*np.sin(pitch), zhub + r]]]

    # Turbine centers x, y, z at 1D upstream, rotor plane, 1D downstream
    # then 3D downstream, 5D downstream, 7D downstream
    # First upwind turbine, then downwind turbine
    turb_centers_frontview = (
            (1134.964, 998.262, zhub),  # S
            (882.964, 1434.738, zhub),  # N
            (1244.083, 1061.262, zhub),  # S
            (992.083, 1497.738, zhub),  # N
            (1353.202, 1124.262, zhub),  # S
            (1101.202, 1560.738, zhub),  # N
            (1571.44, 1250.262, zhub),  # S
            (1319.44, 1686.738, zhub),  # N
            (1789.679, 1376.262, zhub),  # S
            (1537.679, 1812.738, zhub),  # N
            (2007.917, 1502.262, zhub),  # S
            (1755.917, 1938.738, zhub)  # N
            # # Northern turbine
            # (882.964, 1434.738, zhub),
            # (992.083, 1497.738, zhub),
            # (1101.202, 1560.738, zhub),
            # (1319.44, 1686.738, zhub),
            # (1537.679, 1812.738, zhub),
            # (1755.917, 1938.738, zhub)
    )
    # Confine box for plotting only the interesting region
    if 'vert' in slice_orient:
        # 1D upstream, rotor plane, 1D downstream, 3D downstream, 5D downstream, 7D downstream:
        # xmin, xmax, ymin, ymax, zmin, zmax
        confinebox = ((788.464, 1229.464, 834.583, 1598.417, 0., 216.),
                      (897.583, 1338.583, 897.583, 1661.417, 0., 216.),
                      (1006.702, 1447.702, 960.583, 1724.417, 0., 216.),
                      (1224.94, 1665.94, 1086.583, 1850.417, 0., 216.),
                      (1443.179, 1884.179, 1212.583, 1976.417, 0., 216.),
                      (1661.417, 2102.417, 1338.583, 2102.417, 0., 216.))

        # 3D, 5D, 7D downstream: xmin, xmax, ymin, ymax, zmin, zmax
        confinebox2 = ((1224.94, 1665.94, 1086.583, 1850.417, 0., 216.),
                       (1443.179, 1884.179, 1212.583, 1976.417, 0., 216.),
                       (1661.417, 2102.417, 1338.583, 2102.417, 0., 216.))
    else:
        if not rotate_field:
            confinebox = ((950., 2500., 850., 2400., 0., 216.),)*10
        else:
            confinebox = ((994.344*np.cos(rot_z) + 989.821*np.sin(rot_z),
                           994.344*np.cos(rot_z) + 989.821*np.sin(rot_z) + 1764.,
                           989.821*np.cos(rot_z) - 994.344*np.sin(rot_z) - 693,
                           989.821*np.cos(rot_z) - 994.344*np.sin(rot_z) + 1764 - 693, 0., 216.),)*10

        confinebox2 = confinebox

    return turb_borders, turb_centers_frontview, confinebox, confinebox2


def ParTurb_Yaw(slice_orient='vertical', r=63., pitch=5., zhub=90., rot_z=np.pi/6., rotate_field=True):
    # Northern border x, y, southern border x, y of southern turbine
    # then northern border x, y, southern border x, y of northern turbine
    if 'hor' in slice_orient:
        if not rotate_field:
            turb_borders = [[1212.583, 1115.821, 1275.583, 1006.702],
                            [960.583, 1552.298, 1023.583, 1443.179]]
        else:
            turb_borders = [[1212.583*np.cos(rot_z) + np.sin(rot_z)*1115.821 + r*np.sin(1/18.*np.pi), 1115.821*np.cos(rot_z) - 1212.583*np.sin(rot_z) - r*np.sin(1/36.*np.pi),
                             1275.583*np.cos(rot_z) + np.sin(rot_z)*1006.702 - r*np.sin(1/18.*np.pi), 1006.702*np.cos(rot_z) - 1275.583*np.sin(rot_z) + r*np.sin(1/36.*np.pi)],
                            [960.583*np.cos(rot_z) + np.sin(rot_z)*1552.298 + r*np.sin(2/18.*np.pi), 1552.298*np.cos(rot_z) - 960.583*np.sin(rot_z) - r*np.sin(1/18.*np.pi),
                             1023.583*np.cos(rot_z) + 1443.179*np.sin(rot_z) - r*np.sin(2/18.*np.pi), 1443.179*np.cos(rot_z) - 1023.583*np.sin(rot_z) + r*np.sin(1/18.*np.pi)]]

    # Southern turbine then northern turbine: lower border x, z; upper border x, z
    else:
        turb_borders = [[1244.083 - r*np.sin(pitch/180.*np.pi), zhub - r, 1244.083 + r*np.sin(pitch), zhub + r],
                        [[992.083 - r*np.sin(pitch/180.*np.pi), zhub - r, 992.083 + r*np.sin(pitch), zhub + r]]]

    # Turbine centers x, y, z at 1D upstream, rotor plane, 1D downstream
    # then 3D downstream, 5D downstream, 7D downstream
    # First southern turbine, then northern turbine
    turb_centers_frontview = (
            (1134.964, 998.262, zhub),  # S
            (882.964, 1434.738, zhub),  # N
            (1244.083, 1061.262, zhub),  # S
            (992.083, 1497.738, zhub),  # N
            (1353.202, 1124.262, zhub),  # S
            (1101.202, 1560.738, zhub),  # N
            (1571.44, 1250.262, zhub),  # S
            (1319.44, 1686.738, zhub),  # N
            (1789.679, 1376.262, zhub),  # S
            (1537.679, 1812.738, zhub),  # N
            (2007.917, 1502.262, zhub),  # S
            (1755.917, 1938.738, zhub)  # N
            # # Northern turbine
            # (882.964, 1434.738, zhub),
            # (992.083, 1497.738, zhub),
            # (1101.202, 1560.738, zhub),
            # (1319.44, 1686.738, zhub),
            # (1537.679, 1812.738, zhub),
            # (1755.917, 1938.738, zhub)
    )
    # Confine box for plotting only the interesting region
    if 'vert' in slice_orient:
        # 1D upstream, rotor plane, 1D downstream, 3D downstream, 5D downstream, 7D downstream:
        # xmin, xmax, ymin, ymax, zmin, zmax
        confinebox = ((788.464, 1229.464, 834.583, 1598.417, 0., 216.),
                      (897.583, 1338.583, 897.583, 1661.417, 0., 216.),
                      (1006.702, 1447.702, 960.583, 1724.417, 0., 216.),
                      (1224.94, 1665.94, 1086.583, 1850.417, 0., 216.),
                      (1443.179, 1884.179, 1212.583, 1976.417, 0., 216.),
                      (1661.417, 2102.417, 1338.583, 2102.417, 0., 216.))

        # 3D, 5D, 7D downstream: xmin, xmax, ymin, ymax, zmin, zmax
        confinebox2 = ((1224.94, 1665.94, 1086.583, 1850.417, 0., 216.),
                       (1443.179, 1884.179, 1212.583, 1976.417, 0., 216.),
                       (1661.417, 2102.417, 1338.583, 2102.417, 0., 216.))
    else:
        if rotate_field:
            confinebox = ((1120.344*np.cos(rot_z) + 771.583*np.sin(rot_z),
                           882. + 1120.344*np.cos(rot_z) + 771.583*np.sin(rot_z),
                           -1120.344*np.sin(rot_z) + 771.583*np.cos(rot_z),
                           882 - 1120.344*np.sin(rot_z) + 771.583*np.cos(rot_z), 0., 216.),)*10
        else:
            confinebox = ((800., 2400., 800., 2400., 0., 216.),
                          (800., 2400., 800., 2400., 0., 216.))

        confinebox2 = confinebox

    return turb_borders, turb_centers_frontview, confinebox, confinebox2
