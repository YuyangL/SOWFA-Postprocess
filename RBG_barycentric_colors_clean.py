import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def get_barycentric_coords(tensor):
    """
    :param tensor: 3x3 anisotropic Reynolds stress tensor
    :return: x and y barycentric coordinates
    """
    # Compute barycentric coordinates from the eigenvalues of a 3x3 matrix
    eigenvalues_RST = np.linalg.eigvals(tensor)
    eigenvalues_RST = np.flip(np.sort(eigenvalues_RST))
    # Barycentric coordinates
    c1 = eigenvalues_RST[0] - eigenvalues_RST[1]
    c2 = 2. * (eigenvalues_RST[1] - eigenvalues_RST[2])
    c3 = 3. * eigenvalues_RST[2] + 1.

    check_bary = c1 + c2 + c3  # should sum up to 1

    # Define the corners
    xi1 = 1.2
    eta1 = -np.sqrt(3.) / 2.
    xi2 = -0.8
    eta2 = -np.sqrt(3.) / 2.
    xi3 = 0.2
    eta3 = np.sqrt(3.) / 2.

    x_bary = c1 * xi1 + c2 * xi2 + c3 * xi3
    y_bary = c1 * eta1 + c2 * eta2 + c3 * eta3

    return x_bary, y_bary, eigenvalues_RST


def get_barycentric_coords2(tensor):
    """
    :param tensor: 3x3 anisotropic Reynolds stress tensor
    :return: x and y barycentric coordinates, new realizable tensor
    """
    # Compute barycentric coordinates from the eigenvalues of a 3x3 matrix
    eigenvalues_RST_unsorted = np.linalg.eigvals(tensor)
    # Sort eigenvalues based on magnitude
    eig_sorted = np.flip(np.argsort(eigenvalues_RST_unsorted))
    eigenvalues_RST = eigenvalues_RST_unsorted[eig_sorted]
    # Barycentric coordinates
    c1 = eigenvalues_RST[0] - eigenvalues_RST[1]
    c2 = 2. * (eigenvalues_RST[1] - eigenvalues_RST[2])
    c3 = 3. * eigenvalues_RST[2] + 1.

    check_bary = c1 + c2 + c3  # should sum up to 1

    # Define the corners
    xi1 = 1.2
    eta1 = -np.sqrt(3.) / 2.
    xi2 = -0.8
    eta2 = -np.sqrt(3.) / 2.
    xi3 = 0.2
    eta3 = np.sqrt(3.) / 2.

    x_bary = c1 * xi1 + c2 * xi2 + c3 * xi3
    y_bary = c1 * eta1 + c2 * eta2 + c3 * eta3

    if y_bary < eta1:
        # print('\t apply realizability filter')
        x_new = x_bary + (xi3 - x_bary) / (eta3 - y_bary) * (eta1 - y_bary)
        y_new = eta1
        # Solve linear system a beta = y
        a = np.array([[xi1, -xi1 + 2*xi2, -2*xi2+3*xi3],
                      [eta1, -eta1 + 2*eta2, -2*eta2+3*eta3],
                      [1., 1., 1.]])
        y = np.array([[x_new - xi3], [y_new - eta3], [0.]])

        bary_realizable = np.reshape(np.linalg.solve(a, y), [3])

        c1 = bary_realizable[0] - bary_realizable[1]
        c2 = 2. * (bary_realizable[1] - bary_realizable[2])
        c3 = 3. * bary_realizable[2] + 1.

        check_bary = c1 + c2 + c3  # should sum up to 1

        # Define the corners
        xi1 = 1.2
        eta1 = -np.sqrt(3.) / 2.
        xi2 = -0.8
        eta2 = -np.sqrt(3.) / 2.
        xi3 = 0.2
        eta3 = np.sqrt(3.) / 2.

        x_bary = c1 * xi1 + c2 * xi2 + c3 * xi3
        y_bary = c1 * eta1 + c2 * eta2 + c3 * eta3

    else:
        # Enforce general realizability -> sum eigenvalues = 0
        # Solve linear system a beta = y
        a = np.array([[xi1, -xi1 + 2 * xi2, -2 * xi2 + 3 * xi3],
                      [eta1, -eta1 + 2 * eta2, -2 * eta2 + 3 * eta3],
                      [1., 1., 1.]])
        y = np.array([[x_bary - xi3], [y_bary - eta3], [0.]])

        bary_realizable = np.reshape(np.linalg.solve(a, y), [3])

    # Compute new tensor with the realizable barycentric coordinates and eigenvalues
    eig_sorted_reverse = []
    for i in eigenvalues_RST_unsorted:
        eig_sorted_reverse.append(np.where(eigenvalues_RST == i)[0][0])
    tensor_eigvect = np.linalg.eig(tensor)[1]
    vectors = np.vstack([tensor_eigvect[0], tensor_eigvect[1], tensor_eigvect[2]])
    bary_realizable_sorted = bary_realizable[eig_sorted_reverse]
    tensor_new = np.dot(vectors, np.dot(np.diag(bary_realizable_sorted), np.linalg.inv(vectors)))

    return x_bary, y_bary, tensor_new


def get_barycentric_color(x_bary, y_bary):

    # Define the corners
    xi1 = 1.2
    eta1 = -np.sqrt(3.) / 2.
    xi2 = -0.8
    eta2 = -np.sqrt(3.) / 2.
    xi3 = 0.2
    eta3 = np.sqrt(3.) / 2.

    # Set color range and colormap
    steps = 900
    phi_range = np.linspace(0, -2. * np.pi, steps)
    norm = plt.Normalize()
    colors = plt.cm.hsv(norm(phi_range))

    # Determine centroid of barycentric map in [x, y]
    centroid = [xi3, eta3 - (xi1 - xi2) * np.sqrt(3.) / 3.]

    # Determine polar coordinates of the input bary x and y
    radius = np.sqrt((x_bary - centroid[0])**2 + (y_bary - centroid[1])**2)
    delta_phi_1C = np.arctan2(eta1 - centroid[1], xi1 - centroid[0])
    phi = np.arctan2(y_bary - centroid[1], x_bary - centroid[0])

    # Correct for angles in top half of bary map
    if phi >= 0.:
        phi = - (2. * np.pi - phi)

    # set phi zero equal to the anlge of the 1C corner
    phi = phi - delta_phi_1C


    # Correct for angles between phi= 0 and the 1C corner
    if phi >= 0.:
        phi = - (2. * np.pi - phi)

    color_index = steps - np.searchsorted(np.flip(phi_range), phi, side="left") - 1

    # Determine reference radius
    if -120./180. * np.pi < phi <= 0.:
        lhs = np.array([[(y_bary - centroid[1]) / (x_bary - centroid[0]), -1.],
                        [(eta1 - eta2) / (xi1 - xi2), -1.]])
        rhs = np.array([[-centroid[1] + centroid[0] * (y_bary - centroid[1]) / (x_bary - centroid[0])],
                       [-eta2 + xi2 * (eta1 - eta2) / (xi1 - xi2)]])
        coords_side = np.linalg.solve(lhs, rhs)
        max_radius = np.sqrt((coords_side[0] - centroid[0])**2 + (coords_side[1] - centroid[1])**2)
    elif -240./180. * np.pi < phi <= -120./180. * np.pi:
        lhs = np.array([[(y_bary - centroid[1]) / (x_bary - centroid[0]), -1.],
                        [(eta3 - eta2) / (xi3 - xi2), -1.]])
        rhs = np.array([[-centroid[1] + centroid[0] * (y_bary - centroid[1]) / (x_bary - centroid[0])],
                        [-eta2 + xi2 * (eta3 - eta2) / (xi3 - xi2)]])
        coords_side = np.linalg.solve(lhs, rhs)
        max_radius = np.sqrt((coords_side[0] - centroid[0]) ** 2 + (coords_side[1] - centroid[1]) ** 2)
    else:
        lhs = np.array([[(y_bary - centroid[1]) / (x_bary - centroid[0]), -1.],
                        [(eta1 - eta3) / (xi1 - xi3), -1.]])
        rhs = np.array([[-centroid[1] + centroid[0] * (y_bary - centroid[1]) / (x_bary - centroid[0])],
                        [-eta3 + xi3 * (eta1 - eta3) / (xi1 - xi3)]])
        coords_side = np.linalg.solve(lhs, rhs)
        max_radius = np.sqrt((coords_side[0] - centroid[0]) ** 2 + (coords_side[1] - centroid[1]) ** 2)

    # Select color
    bary_colors = colors[color_index, :]

    if radius / max_radius < 1.0:
        bary_colors[3] *= (radius / max_radius)**(1./3)
    else:
        max_radius = (eta3 - centroid[1])
        problem = radius / (eta3 - centroid[1])
        # print('Radius outside barycentric map')

    # Return colors as [R, B, G, alpha], because that is what matplotlib needs
    return bary_colors[[0, 2, 1, 3]]


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def isInside(x1, y1, x2, y2, x3, y3, x, y):
    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)

    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)

    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    if (A == A1 + A2 + A3):
        return True
    else:
        return False

# Plot a sample of the barycentric map
plot_map = True

if plot_map:
    # Sample barycentric map
    # Define the corners
    xi1 = 1.2
    eta1 = -np.sqrt(3.) / 2.
    xi2 = -0.8
    eta2 = -np.sqrt(3.) / 2.
    xi3 = 0.2
    eta3 = np.sqrt(3.) / 2.

    x_sample = np.random.uniform(xi2, xi1, 10000)
    y_sample = np.random.uniform(eta1, eta3, 10000)

    xy_bary = []

    for i in range(1000):
        check_inside = isInside(xi2, eta2, xi3, eta3, xi1, eta1, x_sample[i], y_sample[i])
        if check_inside:
            xy_bary.append([x_sample[i], y_sample[i]])

    # add the corners
    xy_bary.append([xi1, eta1])
    xy_bary.append([xi2, eta2])
    xy_bary.append([xi3, eta3])

    res = []

    for i in range(len(xy_bary)):
        color_i = get_barycentric_color(xy_bary[i][0], xy_bary[i][1])
        res.append(color_i)

    res = np.array(res)

    plt.figure(dpi=150)

    points = np.fliplr(np.array(xy_bary))

    points[:, 0] += 0.8 - (eta3 - 0.8)
    points[:, 1] += eta3

    points /= 2.

    grid_x, grid_y = np.mgrid[0:1:1000j, 0:1:1000j]

    grid_z2 = griddata(points, res, (grid_x, grid_y), method='cubic')

    grid_z2 = np.nan_to_num(grid_z2)

    plt.imshow(grid_z2, extent=(0, 1, 0, 1), origin='lower')

    plt.show()
else:

    # Define training features and responses
    storage_filepath = 'J:/ALM_N_H_ParTurb/Slices/Result/22000.0918025/'
    propertyName = 'uuPrime2'
    sliceName = 'alongWindRotorOne'

    y_test = pickle.load(open(storage_filepath + propertyName + '_' + sliceName + '_tensors.p', 'rb'))
    y_test = np.reshape(y_test, [y_test.shape[0]*y_test.shape[1], 9])

    x = pickle.load(open(storage_filepath + propertyName + '_' + sliceName + '_x.p', 'rb'))
    y = pickle.load(open(storage_filepath + propertyName + '_' + sliceName + '_y.p', 'rb'))
    z = pickle.load(open(storage_filepath + propertyName + '_' + sliceName + '_z.p', 'rb'))

    # meshRANS = pickle.load(open(storage_filepath + '/meshRANS.p', 'rb'))

    # # Get y, z coordinates
    # meshRANS = meshRANS[0:2]
    meshRANS = np.array([x, z])

    # Storage for color data
    res = []

    # Determine if to ensure realizability in barycentric map or not
    ensure_realizability = False

    # Loop over all data in the mesh
    # Check if data is there, otherwise set color to white
    for i in range(y_test.shape[0]):
        if all(np.isfinite(y_test[i, :])):
            # get barycentric coordinates with
            if ensure_realizability:
                x_bary, y_bary, _ = get_barycentric_coords2(np.reshape(y_test[i, :], [3, 3]))
            else:
                x_bary, y_bary, evalues = get_barycentric_coords(np.reshape(y_test[i, :], [3, 3]))
            # determine the associated color
            color_i = get_barycentric_color(x_bary, y_bary)
        else:
            color_i = np.array([0, 0, 0, 0])
        # Store the color
        res.append(color_i)

    # Revert color list to np.array
    res = np.array(res)

    # Plot the result
    fig1 = plt.figure()
    # Stack mash to points (x, y)
    points = np.hstack([np.reshape(meshRANS[1], [y_test.shape[0], 1]), np.reshape(meshRANS[0], [y_test.shape[0], 1])])
    # Create grid used for the final iamge
    grid_x, grid_y = np.mgrid[0:1:3000j, 0:1:1000j]
    # Resize image to fit the points of the flow meshgrid
    grid_x *= (meshRANS[1].max() - meshRANS[1].min())
    grid_y *= (meshRANS[0].max() - meshRANS[0].min())

    grid_x += meshRANS[1].min()
    grid_y += meshRANS[0].min()

    # Interpolate data to image
    grid_z2 = griddata(points, res, (grid_x, grid_y), method='cubic')

    # Resolve any problems with nan values
    grid_z2 = np.nan_to_num(grid_z2)

    # Plot final image
    ax = fig1.add_subplot(1, 1, 1)
    ax.imshow(grid_z2, extent=(meshRANS[0].min(), meshRANS[0].max(), meshRANS[1].min(), meshRANS[1].max()),
              origin='lower')

    plt.savefig('J:/ALM_N_H_ParTurb/Slices/Result/savefig1.png', dpi=600)
    plt.show()

print('Done')
