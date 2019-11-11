# # Create the data.
# from numpy import pi, sin, cos, mgrid
# dphi, dtheta = pi/250.0, pi/250.0
# [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
# m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
# r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
# x = r*sin(phi)*cos(theta)
# y = r*cos(phi)
# z = r*sin(phi)*sin(theta)
#
# # View it.
from mayavi import mlab
from mayavi.modules.contour_grid_plane import ContourGridPlane
from mayavi.modules.outline import Outline
from mayavi.api import Engine
# s = mlab.mesh(x, y, z)
# mlab.show()


from FieldData import FieldData
import numpy as np
# from math import sqrt

caseDir, caseName = 'J:', 'ALM_N_H_ParTurb'
times = '22000.0918025'

fields = FieldData(['U', 'Q'], caseDir = caseDir, caseName = caseName, times = times)

fieldData = fields.readFieldData()
U = fieldData['U']
T = fieldData['Q']

ccx, ccy, ccz, cc = fields.readCellCenterCoordinates()

meshSize, cellSizeMin, ccx3D, ccy3D, ccz3D = fields.getMeshInfo(ccx, ccy, ccz)






# Uhor, w = np.zeros((U.shape[0], 1)), np.zeros((U.shape[0], 1))
# for i, row in enumerate(U):
#     Uhor[i] = np.sqrt(row[0]**2 + row[1]**2)
#     w[i] = row[2]
#
# # Uhor = U[:, 0]
#
# # UhorMean = Uhor.mean()
# UhorMean = 8.  # 6.93
#
# UhorMesh = Uhor.reshape((meshSize[2], meshSize[1], meshSize[0])).T
# UhorMeans = np.zeros((UhorMesh.shape[2], 1))
# UhorpMesh =UhorMesh
# for i in range(UhorMesh.shape[2]):
#     UhorMeans[i] = UhorMesh[:, :, i].mean()
#     UhorpMesh[:, :, i] -= UhorMeans[i]
#
# wMesh = w.reshape((meshSize[2], meshSize[1], meshSize[0])).T
# Tmesh = T.reshape((meshSize[2], meshSize[1], meshSize[0])).T
# # UhorpMesh = UhorMesh - UhorMean





T3D = fields.convertScalarFieldToMeshGrid(T, meshSize)
u, v, w, Uhor = fields.decomposedFields(U)
u3D, v3D, w3D, Uhor3D = fields.convertScalarFieldToMeshGrid(u, meshSize), \
                        fields.convertScalarFieldToMeshGrid(v, meshSize), \
                        fields.convertScalarFieldToMeshGrid(w, meshSize), \
                        fields.convertScalarFieldToMeshGrid(Uhor, meshSize)

UhorRes3D, wRes3D, _, _= fields.getPlanerFluctuations(Uhor3D, w3D)



from PlottingTool_Old import plotIsosurfaces3D
plotIsosurfaces3D(ccx3D, ccy3D, ccz3D, [UhorRes3D, wRes3D], contourList = [[-1.25], [1.]], slice3Dlist = [T3D],
                  boundSurface = (0, 3000, 0, 3000, 0, 500), boundSlice = (0, 3000, 0, 3000, 0, 1000),
                  customColors = [(60/255., 200/255., 255/255.), (244/255., 66/255., 66/255.), 'gist_gray'],
                  sliceOffsets = (20,), sliceValRange = (295, 310), name = caseName, figDir = fields.resultPath)





# engine = Engine()
# engine.start()
# # if len(engine.scenes) == 0:
# #     engine.new_scene()
#
# mlab.figure('Isosurface', engine = engine, size = (1000, 800), bgcolor = (1, 1, 1), fgcolor = (0.5, 0.5, 0.5))
#
# image_plane_widget = mlab.volume_slice(ccx3D, ccy3D, ccz3D, Tmesh, plane_orientation='z_axes', colormap = 'Greys',
#                                        vmin = 298, vmax = 302)
# image_plane_widget.ipw.reslice_interpolate = 'cubic'
# image_plane_widget.ipw.origin = np.array([ 0.,  0., 20.])
# image_plane_widget.ipw.point1 = np.array([3000.,    0.,   20.])
# image_plane_widget.ipw.point2 = np.array([   0., 3000.,   20.])
# # image_plane_widget.ipw.slice_index = 2
# image_plane_widget.ipw.slice_position = 20.0
#
# # cgp = ContourGridPlane()
# # engine.add_module(cgp)
# # cgp.grid_plane.axis = 'y'
# # cgp.grid_plane.position = ccx3D.shape[1] - 1
# # # contour_grid_plane2.actor.mapper.scalar_range = array([298., 302.])
# # # contour_grid_plane2.actor.mapper.progress = 1.0
# # # contour_grid_plane2.actor.mapper.scalar_mode = 'use_cell_data'
# # cgp.contour.filled_contours = True
#
# mlab.contour3d(ccx3D, ccy3D, ccz3D, wMesh, contours = [1.], color = (244/255., 66/255., 66/255.), extent = [0,
#                                                                                                                3000, 0, 3000,20, 500])
#
# mlab.contour3d(ccx3D, ccy3D, ccz3D, UhorpMesh, contours = [-1.25], color = (60/255., 200/255., 255/255.), extent = [0,
#                                                                                                                  3000, 0, 3000,
#                                                                                                     20, 500])
#
# # mlab.axes(figure = myfig, x_axis_visibility = True, y_axis_visibility = True, z_axis_visibility = True, ranges = [0,
# #                                                                                                                  3000, 0, 3000,
# #                                                                                                   0, 500])
#
# outline = Outline()
# engine.add_module(outline)
# outline.actor.property.color = (0.3, 0.3, 0.3)
# outline.bounds = np.array([  0,    3000,   0,    3000,   0, 500.])
# outline.manual_bounds = True
#
# # field = mlab.pipeline.scalar_field(ccx3D, ccy3D, ccz3D, UhorMesh)
# # mlab.pipeline.iso_surface(mlab.pipeline.extract_vector_norm(field))
# mlab.view(azimuth = 270, elevation = 45)
# # mlab.savefig('isosurface.png', magnification = 3)
# mlab.show()


