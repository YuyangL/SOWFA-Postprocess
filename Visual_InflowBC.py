import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

# Pretty much defines which time to plot
# skip_header = 2700
max_rows = 1

case = 'N_H'

if case == 'N_H':
    caseSetting = {'caseDir': 'ABL_N_H/12826.8494412/',
           'zInv': 745,
           'eqTime': 19826.8494412}
elif case == 'N_L':
    caseSetting = {'caseDir': 'ABL_N_L/15738.0644797/',
           'zInv': 755,
           'eqTime': 19738.0644797}
elif case == 'U_H':
    caseSetting = {'caseDir': 'ABL_Uns_H/0/',
           'zInv': 765,
           'eqTime': 0.}
elif case == 'U_L':
    caseSetting = {'caseDir': 'ABL_Uns_L/0/',
           'zInv': 745,
           'eqTime': 0.}

# ABL_N_H/12826.8494412/
# ABL_N_L/15738.0644797/
caseDir = caseSetting['caseDir']

heights = np.genfromtxt(caseDir + '/hLevelsCell')


zHub = 90
# For ABL_N_H: 745
# For ABL_N_L: 755
zInv = caseSetting['zInv']

# For ABL_N_H: 19826.8494412
# For ABL_N_L: 19738.0644797
targetEqTime = caseSetting['eqTime']

# In case zHub is not in the list of heights
try:
    zloc = np.where(heights == zHub)[0][0]
except IndexError:
    zloc = np.where(heights == zHub - 5)[0][0]


"""
Velocity
"""
def velocity(sub = '', skip_header = 0, targetEqTime = 0.):
    # If no target equilibruim time is provided, use skip_header request
    if targetEqTime == 0:
        data_U = np.genfromtxt(caseDir + 'U_mean' + sub, skip_header = skip_header, max_rows = max_rows)
        data_V = np.genfromtxt(caseDir + 'V_mean' + sub, skip_header = skip_header, max_rows = max_rows)
        # NO SUB YET
        data_W = np.genfromtxt(caseDir + 'W_mean', skip_header = 1, max_rows = max_rows)
    else:
        data_U = np.genfromtxt(caseDir + 'U_mean' + sub)
        data_V = np.genfromtxt(caseDir + 'V_mean' + sub)
        # NO SUB YET
        data_W = np.genfromtxt(caseDir + 'W_mean')

        data_U = data_U[np.where(data_U[:, 0] == targetEqTime)].flatten()
        data_V = data_V[np.where(data_V[:, 0] == targetEqTime)].flatten()
        data_W = data_W[np.where(data_W[:, 0] == targetEqTime)].flatten()


    ts = np.genfromtxt(caseDir + 'U_mean' + sub)[:, 0]

    # Trim the time col and the dt col
    data_U, data_V, data_W = data_U[2:], data_V[2:], data_W[2:]

    Vel_hor = np.sqrt(np.square(data_U) + np.square(data_V))

    Vel = np.sqrt(np.square(data_U) + np.square(data_V) + np.square(data_W))

    Vel_hor_hub = Vel_hor[zloc]

    Vel_hub = Vel[zloc]
    return Vel, Vel_hor, Vel_hor_hub, Vel_hub, ts


Vel, Vel_hor, Vel_hor_hub, Vel_hub, ts = velocity(targetEqTime = targetEqTime)


"""
Temperature
"""
def temperature(sub = '', skip_header = 0, targetEqTime = 0.):
    # If no target equilibrium time is provided, use skip_header request
    if targetEqTime == 0:
        data_T = np.genfromtxt(caseDir + 'T_mean' + sub, skip_header = skip_header, max_rows = max_rows)
    else:
        data_T = np.genfromtxt(caseDir + 'T_mean' + sub)
        data_T = data_T[np.where(data_T[:, 0] == targetEqTime)].flatten()

    data_T = data_T[2:]
    return data_T


T = temperature(targetEqTime = targetEqTime)


"""
Turbulence Intensity
TI = u'(z)/U(z)
"""
def turbulentIntensity(Vel, sub = '', skip_header = 0, targetEqTime = 0.):
    # If no target equilibrium time is provided, use skip_header request
    if targetEqTime == 0:
        data_uu = np.genfromtxt(caseDir + 'uu_mean' + sub, skip_header = skip_header, max_rows = max_rows)
        # NO SUB YET
        data_vv = np.genfromtxt(caseDir + 'vv_mean', skip_header = 1, max_rows = max_rows)
        data_ww = np.genfromtxt(caseDir + 'ww_mean', skip_header = 1, max_rows = max_rows)
    else:
        data_uu = np.genfromtxt(caseDir + 'uu_mean' + sub)
        # NO SUB YET
        data_vv = np.genfromtxt(caseDir + 'vv_mean')
        data_ww = np.genfromtxt(caseDir + 'ww_mean')

        data_uu = data_uu[np.where(data_uu[:, 0] == targetEqTime)].flatten()
        data_vv = data_vv[np.where(data_vv[:, 0] == targetEqTime)].flatten()
        data_ww = data_ww[np.where(data_ww[:, 0] == targetEqTime)].flatten()

    data_uu, data_vv, data_ww = data_uu[2:], data_vv[2:], data_ww[2:]

    TI = np.divide(np.sqrt((data_uu + data_vv + data_ww)/3), Vel)
    # TI = np.divide(np.sqrt(data_uu), Vel_hor)
    return TI


TI = turbulentIntensity(Vel, targetEqTime = targetEqTime)


def verticleTempFlux():
    data_q = np.genfromtxt(caseDir + 'q3_mean', skip_header = 100, max_rows = 1)

    data_q = data_q[2:]

    return np.abs(data_q)


q = verticleTempFlux()

q_min_loc = np.argmin(q)

plt.plot(q, heights)

plt.show()



"""
Processing
"""


# 
# step0 = 0
# step1 = 2000
# step2 = 1000
# [Vel_hor0, Vel_hor_hub10, Vel_hor_hub20, ts0] = velocity('0', step0)
# [Vel_hor1, Vel_hor_hub11, Vel_hor_hub21, ts1] = velocity('0', step1)
# [Vel_hor2, Vel_hor_hub12, Vel_hor_hub22, ts2] = velocity('', step2)
# 
# T0 = temperature('0', step0)
# T1 = temperature('0', step1)
# T2 = temperature('', step2)
# 
# TI0, TI1, TI2 = TI(Vel_hor0, '0', step0), TI(Vel_hor1, '0', step1), TI(Vel_hor2, '', step2)
# 
"""
Plots
"""
# # Velocity
# plt.figure('Inflow velocity profile')
# plt.plot(Vel/Vel_hub, heights/zInv)
# # plt.plot(Vel_hor1/Vel_hor_hub21, heights/zHub, label = 't = %d s' % ts1[step1])
# # plt.plot(Vel_hor2/Vel_hor_hub22, heights/zHub, label = 't = %d s' % ts2[step2])
# # plt.legend()
# plt.xlabel(r'$<U>/U_{Hub}$')
# plt.ylabel(r'$z/z_{i}$')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.grid()
# #
# # Temperature
# plt.figure('Inflow temperature profile')
# plt.plot(T, heights/zHub)
# # plt.plot(T0, heights/zHub, label = 't = %d s' % ts0[step0])
# # plt.plot(T1, heights/zHub, label = 't = %d s' % ts1[step1])
# # plt.plot(T2, heights/zHub, label = 't = %d s' % ts2[step2])
# # plt.legend()
# plt.xlabel('<T> [k]')
# plt.ylabel(r'$z/z_{Hub}$')
# plt.grid()
# #
# # TI
# plt.figure('Inflow TI profile')
# plt.plot(TI, heights/zHub)
# # plt.plot(TI0, heights/zHub, label = 't = %d s' % ts0[step0])
# # plt.plot(TI1, heights/zHub, label = 't = %d s' % ts1[step1])
# # plt.plot(TI2, heights/zHub, label = 't = %d s' % ts2[step2])
# # plt.legend()
# plt.xlabel('I')
# plt.ylabel(r'$z/z_{Hub}$')
# plt.grid()
# #
# plt.show()
