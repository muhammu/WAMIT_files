import numpy as np


class LO_mesh(object):
    def __init__(self):
        """
        WAMIT low-order mesh class.
        """
        self.origo = None
        self.type = None
        self.symmetry = None
        self.wetside = None
        self.panelmat = None


def to_rosetta_format(panel_mat):
    """
    SHORT DESCRIPTION:
    modify the generated vertex matrices into BEMRosetta format

    INPUT:
    panel_mat: i x j matrix
      where:
        i = panel number,
        j = vertex number for each panel (always 12)

    OUTPUT:
    panel_rosetta: k x l matrix
      where:
        k = vertex number,
        l = 3 (i.e., x,y,z)
    #
    Muhammad Mukhlas, Norwegian University of Science and Technology, 2021
    """
    panel_rosetta = np.array([])
    for i in range(0, panel_mat.shape[0]):
        if i == 0:
            panel_rosetta = np.append(panel_rosetta, panel_mat[i, :3])
        else:
            panel_rosetta = np.vstack((panel_rosetta, panel_mat[i, :3]))
        panel_rosetta = np.vstack((panel_rosetta, panel_mat[i, 3:6]))
        panel_rosetta = np.vstack((panel_rosetta, panel_mat[i, 6:9]))
        panel_rosetta = np.vstack((panel_rosetta, panel_mat[i, 9:12]))
    return panel_rosetta


def hemisphere(r, n_hoop, n_merid, symm, wet_side, org=None):
    """
    SHORT DESCRIPTION:
    Generate low-order mesh of a hemisphere.

    INPUT:
    - r: radius of the hemisphere [m]
    - n_hoop: number of panel in hoop direction (consider only quarterth) [-]
    - n_merid: number of panel in meriodional direction [-]
    - org: 1-D numpy array of the origo of the hemisphere (i.e., [xo, yo, zo])
    - symm: symmetry of the body.
            0 = no symmetry,
            1 = symmetry (1/4th of the body)
    - wet_side:
            0 = water is outside of the body,
            1 = water is inside the body

    OUTPUT:
    hemi_panel = i x j matrix
    where:
     i = panel number,
     j = vertex number for each panel (always 12)
    #
    Muhammad Mukhlas, Norwegian University of Science and Technology, 2021
    """

    if org is None:   # if not declared, set [0., 0., 0.] as default
        org = np.array([0., 0., 0.])
    assert wet_side == 0 or wet_side == 1, "please set wet_side = 0 or 1"
    assert symm == 0 or symm == 1, "please set symm = 0 or 1"
    assert r > 0, "please enter r > 0"
    assert n_hoop > 2, "please enter n_hoop > 2"
    assert n_merid > 2, "please enter n_merid > 2"

    # Set the body mesh as an instance
    mesh = LO_mesh()
    mesh.origo = org
    mesh.type = 'hemisphere'
    if wet_side == 0:
        mesh.wetside = 'exterior'
    else:
        mesh.wetside = 'interior'
    if symm == 0:
        mesh.symmetry = 'none'
    else:
        mesh.symmetry = 'ISX_local + ISY_local'

    # Define meridional (theta) and hoop (phi) angles, in radians
    theta = np.linspace(np.pi / 2, np.pi, n_merid + 1)
    if symm == 1:
        phi = np.linspace(np.pi / 2, 0, n_hoop + 1)
    else:
        phi = np.linspace(2 * np.pi, 0, (4 * n_hoop) + 1)

    # Scatter vertices (x[i,j], y[i,j], z[i,j]) around the discretized
    # hemisphere surface. i = no. of vertices in hoop direction,
    #                     j = no. of vertices in meridional direction.
    x_hemi = np.transpose(org[0] + r * np.outer(np.sin(theta), np.cos(phi)))
    y_hemi = np.transpose(org[1] + r * np.outer(np.sin(theta), np.sin(phi)))
    z_hemi = np.transpose(org[2] + r * np.outer(np.cos(theta), np.ones(n_hoop + 1)))

    # Make blank array for each vertex
    v_1_x = np.zeros(0)
    v_1_y = np.zeros(0)
    v_1_z = np.zeros(0)
    #
    v_2_x = np.zeros(0)
    v_2_y = np.zeros(0)
    v_2_z = np.zeros(0)
    #
    v_3_x = np.zeros(0)
    v_3_y = np.zeros(0)
    v_3_z = np.zeros(0)
    #
    v_4_x = np.zeros(0)
    v_4_y = np.zeros(0)
    v_4_z = np.zeros(0)

    # For-loop to define vertex 1-4 for each panel
    for j in range(0, n_merid):
        v_1_x = np.append(v_1_x, x_hemi[:n_hoop, j])
        v_1_y = np.append(v_1_y, y_hemi[:n_hoop, j])
        v_1_z = np.append(v_1_z, z_hemi[:n_hoop, j])
        #
        v_3_x = np.append(v_3_x, x_hemi[1:n_hoop + 1, j + 1])
        v_3_y = np.append(v_3_y, y_hemi[1:n_hoop + 1, j + 1])
        v_3_z = np.append(v_3_z, z_hemi[1:n_hoop + 1, j + 1])
        #
        if wet_side == 0:
            v_2_x = np.append(v_2_x, x_hemi[1:n_hoop + 1, j])
            v_2_y = np.append(v_2_y, y_hemi[1:n_hoop + 1, j])
            v_2_z = np.append(v_2_z, z_hemi[1:n_hoop + 1, j])
            #
            v_4_x = np.append(v_4_x, x_hemi[:n_hoop, j + 1])
            v_4_y = np.append(v_4_y, y_hemi[:n_hoop, j + 1])
            v_4_z = np.append(v_4_z, z_hemi[:n_hoop, j + 1])
        elif wet_side == 1:
            v_2_x = np.append(v_2_x, x_hemi[:n_hoop, j + 1])
            v_2_y = np.append(v_2_y, y_hemi[:n_hoop, j + 1])
            v_2_z = np.append(v_2_z, z_hemi[:n_hoop, j + 1])
            #
            v_4_x = np.append(v_4_x, x_hemi[1:n_hoop + 1, j])
            v_4_y = np.append(v_4_y, y_hemi[1:n_hoop + 1, j])
            v_4_z = np.append(v_4_z, z_hemi[1:n_hoop + 1, j])

    mesh.panelmat = np.transpose(np.array([v_1_x, v_1_y, v_1_z,
                                           v_2_x, v_2_y, v_2_z,
                                           v_3_x, v_3_y, v_3_z,
                                           v_4_x, v_4_y, v_4_z]))
    return mesh


def hollow_cylinder(r_in, r_out, h, n_hoop, n_r, n_h, symm, wet_side, org=None):
    """
    SHORT DESCRIPTION:
    Generate low-order mesh of a hollow cylinder.

    INPUT:
    - r_in: inner radius [m]
    - r_out: outer radius [m]
    - n_r: number of panel along the radius [m]
    - phi: hoop angle [rad]
    - n_phi: number of panel around the hoop angle (consider only quarterth) [-]
    - h: height of the cylinder
    - n_h: number of panel along the height [-]
    - symm: symmetry of the body.
            0 = no symmetry,
            1 = symmetry (1/4th of the body)
    - wet_side:
            0 = water is outside of the body,
            1 = water is inside the body

    OUTPUT:
    hollowcyl_panel = i x j matrix
    where:
     i = panel number,
     j = vertex number for each panel (always 12)

    Muhammad Mukhlas, Norwegian University of Science and Technology, 2021
    """
    if org is None:   # if not declared, set [0., 0., 0.] as default
        org = np.array([0., 0., 0.])
    assert wet_side == 0 or wet_side == 1, "please set wet_side = 0 or 1"
    assert symm == 0 or symm == 1, "please set symm = 0 or 1"
    assert r_in < r_out, "please enter r_in < r_out"
    assert n_hoop > 2, "please enter n_hoop > 2"
    assert n_h > 2, "please enter nh > 2"

    # Define hoop (phi) angles, in radians
    if symm == 1:
        phi = np.linspace(np.pi / 2, 0, n_hoop + 1)
    else:
        phi = np.linspace(2 * np.pi, 0, (4 * n_hoop) + 1)

    # Set the body mesh as an instance
    mesh = LO_mesh()
    mesh.origo = org
    mesh.type = 'hollow cylinder'
    if wet_side == 0:
        mesh.wetside = 'exterior'
    else:
        mesh.wetside = 'interior'
    if symm == 0:
        mesh.symmetry = 'none'
    else:
        mesh.symmetry = 'ISX_local + ISY_local'

    # cylinder base
    r_dir = np.linspace(r_in, r_out, n_r + 1)
    x_base = org[0] + np.outer(r_dir, np.cos(phi))
    y_base = org[1] + np.outer(r_dir, np.sin(phi))
    z_base = org[2] + np.ones(x_base.shape) * (-h)

    # cylinder wall
    x_wall = org[0] + (r_out * np.cos(phi))
    z_wall = org[2] + np.linspace(-h, 0, n_h + 1)
    x_wall, z_wall = np.meshgrid(x_wall, z_wall)
    y_wall = np.sqrt(r_out ** 2 - x_wall ** 2)

    # Make blank array for each vertex
    v_1_x = np.zeros(0)
    v_1_y = np.zeros(0)
    v_1_z = np.zeros(0)
    #
    v_2_x = np.zeros(0)
    v_2_y = np.zeros(0)
    v_2_z = np.zeros(0)
    #
    v_3_x = np.zeros(0)
    v_3_y = np.zeros(0)
    v_3_z = np.zeros(0)
    #
    v_4_x = np.zeros(0)
    v_4_y = np.zeros(0)
    v_4_z = np.zeros(0)

    # For-loop to define vertex 1-4 for each panel as per WAMIT .gdf convention
    # the cylinder base
    for j in range(0, n_r):
        v_1_x = np.append(v_1_x, x_base[j, :n_hoop])
        v_1_y = np.append(v_1_y, y_base[j, :n_hoop])
        v_1_z = np.append(v_1_z, z_base[j, :n_hoop])
        #
        v_3_x = np.append(v_3_x, x_base[j+1, 1:n_hoop + 1])
        v_3_y = np.append(v_3_y, y_base[j+1, 1:n_hoop + 1])
        v_3_z = np.append(v_3_z, z_base[j+1, 1:n_hoop + 1])
        #
        if wet_side == 0:
            v_2_x = np.append(v_2_x, x_base[j + 1, :n_hoop])
            v_2_y = np.append(v_2_y, y_base[j + 1, :n_hoop])
            v_2_z = np.append(v_2_z, z_base[j + 1, :n_hoop])
            #
            v_4_x = np.append(v_4_x, x_base[j, 1:n_hoop + 1])
            v_4_y = np.append(v_4_y, y_base[j, 1:n_hoop + 1])
            v_4_z = np.append(v_4_z, z_base[j, 1:n_hoop + 1])
        #
        elif wet_side == 1:
            v_2_x = np.append(v_2_x, x_base[j, 1:n_hoop + 1])
            v_2_y = np.append(v_2_y, y_base[j, 1:n_hoop + 1])
            v_2_z = np.append(v_2_z, z_base[j, 1:n_hoop + 1])
            #
            v_4_x = np.append(v_4_x, x_base[j + 1, :n_hoop])
            v_4_y = np.append(v_4_y, y_base[j + 1, :n_hoop])
            v_4_z = np.append(v_4_z, z_base[j + 1, :n_hoop])

    # the cylinder wall
    for j in range(0, n_h):
        v_1_x = np.append(v_1_x, x_wall[j, :n_hoop])
        v_1_y = np.append(v_1_y, y_wall[j, :n_hoop])
        v_1_z = np.append(v_1_z, z_wall[j, :n_hoop])
        #
        v_3_x = np.append(v_3_x, x_wall[j+1, 1:n_hoop + 1])
        v_3_y = np.append(v_3_y, y_wall[j+1, 1:n_hoop + 1])
        v_3_z = np.append(v_3_z, z_wall[j+1, 1:n_hoop + 1])
        #
        if wet_side == 0:
            v_2_x = np.append(v_2_x, x_wall[j + 1, :n_hoop])
            v_2_y = np.append(v_2_y, y_wall[j + 1, :n_hoop])
            v_2_z = np.append(v_2_z, z_wall[j + 1, :n_hoop])
            #
            v_4_x = np.append(v_4_x, x_wall[j, 1:n_hoop + 1])
            v_4_y = np.append(v_4_y, y_wall[j, 1:n_hoop + 1])
            v_4_z = np.append(v_4_z, z_wall[j, 1:n_hoop + 1])
        #
        elif wet_side == 1:
            v_2_x = np.append(v_4_x, x_wall[j, 1:n_hoop + 1])
            v_2_y = np.append(v_4_y, y_wall[j, 1:n_hoop + 1])
            v_2_z = np.append(v_4_z, z_wall[j, 1:n_hoop + 1])
            #
            v_4_x = np.append(v_2_x, x_wall[j + 1, :n_hoop])
            v_4_y = np.append(v_2_y, y_wall[j + 1, :n_hoop])
            v_4_z = np.append(v_2_z, z_wall[j + 1, :n_hoop])

    # Array (total panel x 12 coord for vertex 1-4) which it should be okay to use for WAMIT
    mesh.panelmat = np.transpose(np.array([v_1_x, v_1_y, v_1_z,
                                           v_2_x, v_2_y, v_2_z,
                                           v_3_x, v_3_y, v_3_z,
                                           v_4_x, v_4_y, v_4_z]))
    return mesh


def rectangular(B, H, n_B, n_H, symm, wet_side, org=None):
    """
    SHORT DESCRIPTION:
    Generate low-order mesh of a rectangle. Imagine a coordinate systems
    with x pointing towards you, y to your right, and z upwards.

    INPUT:
    - B: breadth of the rectangle [m]
    - H: height of the rectangle [m]
    - n_B: number of panel along the breadth (halfth) [m]
    - n_H: number of panel along the height [rad]
    - symm: symmetry of the body.
            0 = no symmetry,
            1 = symmetry (1/2th of the body)
    - wet_side:
            0 = you see the rectangle from the water,
            1 = you see the rectangle from the dry part.

    OUTPUT:
    rectg_panel = i x j matrix
    where:
     i = panel number,
     j = vertex number for each panel (always 12)

    Muhammad Mukhlas, Norwegian University of Science and Technology, 2021
    """

    if org is None:   # if not declared, set [0., 0., 0.] as default
        org = np.array([0., 0., 0.])
    assert wet_side == 0 or wet_side == 1, "please set wet_side = 0 or 1"
    assert symm == 0 or symm == 1, "please set symm = 0 or 1"

    if symm == 1:
        B_dir = np.linspace(0, B/2, n_B + 1) + org[1]
        H_dir = - np.linspace(0, H, n_H + 1) + org[2]
    elif symm == 0:
        B_dir = np.linspace(-B/2, B/2, (2 * n_B) + 1) + org[1]
        H_dir = - np.linspace(0, H, n_H + 1) + org[2]

    # Set the body mesh as an instance
    mesh = LO_mesh()
    mesh.origo = org
    mesh.type = 'rectangle'
    if wet_side == 0:
        mesh.wetside = 'front'
    else:
        mesh.wetside = 'back'
    if symm == 0:
        mesh.symmetry = 'none'
    else:
        mesh.symmetry = 'ISY_local'

    y_rect, z_rect = np.meshgrid(B_dir, H_dir)
    x_rect = np.zeros_like(y_rect)

    # Make blank array for each vertex
    v_1_x = np.zeros(0)
    v_1_y = np.zeros(0)
    v_1_z = np.zeros(0)
    #
    v_2_x = np.zeros(0)
    v_2_y = np.zeros(0)
    v_2_z = np.zeros(0)
    #
    v_3_x = np.zeros(0)
    v_3_y = np.zeros(0)
    v_3_z = np.zeros(0)
    #
    v_4_x = np.zeros(0)
    v_4_y = np.zeros(0)
    v_4_z = np.zeros(0)
    #
    # For-loop to define vertex 1-4 for each panel as per WAMIT .gdf convention
    for j in range(0, n_H):
        v_1_x = np.append(v_1_x, x_rect[j, :n_B])
        v_1_y = np.append(v_1_y, y_rect[j, :n_B])
        v_1_z = np.append(v_1_z, z_rect[j, :n_B])
        #
        v_3_x = np.append(v_3_x, x_rect[j + 1, 1:n_B + 1])
        v_3_y = np.append(v_3_y, y_rect[j + 1, 1:n_B + 1])
        v_3_z = np.append(v_3_z, z_rect[j + 1, 1:n_B + 1])
        #
        # WAMIT has convention to determine which side is wet and vice versa
        if wet_side == 0:
            v_2_x = np.append(v_2_x, x_rect[j, 1:n_B + 1])
            v_2_y = np.append(v_2_y, y_rect[j, 1:n_B + 1])
            v_2_z = np.append(v_2_z, z_rect[j, 1:n_B + 1])
            #
            v_4_x = np.append(v_4_x, x_rect[j + 1, :n_B])
            v_4_y = np.append(v_4_y, y_rect[j + 1, :n_B])
            v_4_z = np.append(v_4_z, z_rect[j + 1, :n_B])
        elif wet_side == 1:
            v_2_x = np.append(v_2_x, x_rect[j + 1, :n_B])
            v_2_y = np.append(v_2_y, y_rect[j + 1, :n_B])
            v_2_z = np.append(v_2_z, z_rect[j + 1, :n_B])
            #
            v_4_x = np.append(v_4_x, x_rect[j, 1:n_B + 1])
            v_4_y = np.append(v_4_y, y_rect[j, 1:n_B + 1])
            v_4_z = np.append(v_4_z, z_rect[j, 1:n_B + 1])
        #
    # Array (total panel x 12 coord for vertex 1-4) which it should be okay to use for WAMIT
    mesh.panelmat = np.transpose(np.array([v_1_x, v_1_y, v_1_z,
                                           v_2_x, v_2_y, v_2_z,
                                           v_3_x, v_3_y, v_3_z,
                                           v_4_x, v_4_y, v_4_z]))
    return mesh