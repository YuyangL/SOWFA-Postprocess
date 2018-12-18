# data = open("I:/SOWFA Data/ABL_N_H/Field/20000.9038025/U", "rb").read()
# import vtk
# from vtk.util import numpy_support as VN
#
# reader = vtk.vtkPolyDataReader()
# reader.SetFileName('./ABL_N_H/Slices/20060.9038025/U_slice_horizontal_2.vtk')
#
# reader.Update()
#
# data = reader.GetOutput()
#
# u0 = data.GetCellData()
# u = data.GetCellData().GetArray('U')
# u_np = VN.vtk_to_numpy(data.GetCellData().GetArray('U'))
#
#
#
# myArr = vtk.vtkPassArrays()
# myArr.SetInputDataObject(reader.GetInputDataObject(0, 0))



# p2c = vtk.vtkCellDataToPointData()
# p2c.SetInputConnection(reader.GetOutputPort())
# p2c.Update()



import Ofpp as of
import numpy as np
import matplotlib.pyplot as plt
# From low to high x, then low to high y, lastly, low to high z
U = of.parse_internal_field('./ABL_N_H/Field/20000.9038025/U')

ccx = of.parse_internal_field('./ABL_N_H/Field/20000.9038025/ccx')

ccy = of.parse_internal_field('./ABL_N_H/Field/20000.9038025/ccy')

ccz = of.parse_internal_field('./ABL_N_H/Field/20000.9038025/ccz')

cc = np.vstack((ccx, ccy, ccz))

sliceOrigin = (0, 0, 90)
sliceNormal = (0, 0, 1)


idx = np.argwhere((ccz > 94) & (ccz < 96))

Uground = U[0:90000]
Uhub = (U[810000:900000] + U[720000:810000])/2.
U3 = U[90000*21:90000*22]

uGround, vGround, wGround = Uground[:, 0], Uground[:, 1], Uground[:, 2]
uHub, vHub, wHub = Uhub[:, 0], Uhub[:, 1], Uhub[:, 2]
u3, v3, w3 = U3[:, 0], U3[:, 1], U3[:, 2]

uGroundMesh, vGroundMesh, wGroundMesh = uGround.reshape((300, 300)), vGround.reshape((300, 300)), wGround.reshape((300,
                                                                                                               300))
uHubMesh, vHubMesh, wHubMesh = uHub.reshape((300, 300)), vHub.reshape((300, 300)), wHub.reshape((300, 300))
u3Mesh, v3Mesh, w3Mesh = u3.reshape((300, 300)), v3.reshape((300, 300)), w3.reshape((300, 300))

UgroundMagMesh = np.sqrt(np.square(uGroundMesh) + np.square(vGroundMesh) + np.square(wGroundMesh))
UhubMagMesh = np.sqrt(np.square(uHubMesh) + np.square(vHubMesh) + np.square(wHubMesh))
U3MagMesh = np.sqrt(np.square(u3Mesh) + np.square(v3Mesh) + np.square(w3Mesh))

x, y = np.arange(5, 2996, 10), np.arange(5, 2996, 10)
xv, yv = np.meshgrid(x, y)


from PlottingTool import plot2D
plot2D([xv], [yv], z2D = UhubMagMesh, contourLvl = 20, zLabel = '$U$ [m/s]')




fig, ax = plt.subplots()
N = 50
plot = ax.contourf(UhubMagMesh, N, extend = 'both')
cb = fig.colorbar(plot, drawedges = False)
# Remove colorbar outline
cb.outline.set_linewidth(0)
plt.tight_layout()
plt.savefig('2dplot.png', dpi = 1000, transparent = True, bbox_inches = 'tight')


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
fig = plt.figure()
ax2 = fig.gca(projection='3d')
alpha = 0.5
ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([0.65, 0.65, 1.3, 1]))

plot = ax2.contourf(xv, yv, UgroundMagMesh, N, zdir = 'z', offset = 5, alpha = 1)
ax2.contourf(xv, yv, UhubMagMesh, N, zdir = 'z', offset = 90, alpha = 1)
ax2.contourf(xv, yv, U3MagMesh, N, zdir = 'z', offset = 215, alpha = 1)

# Adjust the limits, ticks and view angle
ax2.set_zlim(5, 215)
# ax2.set_zticks(np.linspace(0,0.2,5))
ax2.set_xticks(np.arange(0, 3001, 1000))
ax2.set_yticks(np.arange(0, 3001, 1000))
ax2.view_init(35, -120)
cbaxes = fig.add_axes([0.8, 0.1, 0.02, 0.8])
cb = plt.colorbar(plot, cax = cbaxes, drawedges = False, extend = 'both')
# Remove colorbar outline
cb.outline.set_linewidth(0)
# Turn off background on all three panes
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

plt.tight_layout()
plt.savefig('3dplot.png', dpi = 1000, transparent = True, bbox_inches = 'tight')
plt.show()




# cczHub = ccz[(ccz > 94) & (ccz < 96)]

# ccHub = cc[]

# lst = [cc[:, i] for i, val in enumerate(ccz) if val == 95]
#
# lst2 = cc[np.where(ccz == 95)]
