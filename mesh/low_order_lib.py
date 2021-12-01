import numpy as np


class LO_mesh(object):
    def __init__(self):
        """
        WAMIT low-order mesh class.
        """
        self.origo = None
        self.type = None
        self.ISX = None
        self.ISY = None
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


def cosine_spacing(start_point, end_point, num_points):
    """
    SHORT DESCRIPTION:
    generate cosine spaced vector (1xN), where the spacing concentrates samples
    at the ends while producing fewer sample points in the center.

    INPUT:
        - start_point = index 1 of the vector
        - end_point = index N of the vector
    NOTE THAT END-POINT SHOULD BE GREATER THAN THE START POINT

    OUTPUT:
    x: 1 x N vector

    Muhammad Mukhlas, Norwegian University of Science and Technology, 2021
    """

    assert (end_point > start_point), "End point must be greater than the start point"
    assert (num_points % 2) == 1, "It is recommended to use odd num_points"

    x = np.zeros(num_points)
    x[0] = start_point
    x[-1] = end_point

    mid_point = (end_point - start_point) / 2
    angle_inc = np.pi / (num_points - 1)

    cur_angle = angle_inc
    for idx in range(1, int(1 + ((num_points -1) / 2))):
        x[idx] = start_point + (mid_point * (1 - np.cos(cur_angle)))
        x[-1 - idx] = x[-1 - (idx - 1)] - (x[idx] - x[idx - 1])
        cur_angle += angle_inc
    return x


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
        mesh.ISX = 0
        mesh.ISY = 0
    else:
        mesh.ISX = 1
        mesh.ISY = 1

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
        mesh.ISX = 0
        mesh.ISY = 0
    else:
        mesh.ISX = 1
        mesh.ISY = 1

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
        mesh.ISX = 0
        mesh.ISY = 0
    else:
        mesh.ISX = 0
        mesh.ISY = 1

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


def vertical_membrane_cos(B, H, n_B, th, symm, wet_side, m=1, org=None):
    """
    SHORT DESCRIPTION:
    Generate low-order mesh of a vertical surface penetrating membrane
    with finite thickness th. The mesh spacing along the breadth uses
    the so-called "cosine spacing", i.e. area near the membrane edges
    is more refined.

    INPUT:
    - B: breadth of the membrane [m]
    - H: total submerged height of the membrane [m]
    - m: multiplied of dz (spacing along height), where in default it is equal to 1
    - th: membrane thickness
    - n_B: number of panel along the breadth
    - symm: symmetry of the body.
            0 = no symmetry (HAVE NOT BEEN INCLUDED YET)
            1 = symmetry (1/4th of the body)
    - wet_side:
            0 = wet outside of the membrane
            1 = wet inside of the membrane

    OUTPUT:
    mesh = i x j matrix
    where:
     i = panel number,
     j = vertex number for each panel (always 12)

    Muhammad Mukhlas, Norwegian University of Science and Technology, 2021
    """
    if org is None:   # if not declared, set [0., 0., 0.] as default
        org = np.array([0., 0., 0.])
    assert wet_side == 0 or wet_side == 1, "please set wet_side = 0 or 1"
    assert symm == 1, "please set symm = 1" # TODO: under development

    # number of points in breadth direction
    if (n_B % 2) != 0:
        n_B += 1
    num_points = n_B + 1

    # Generate cosine spacing vector for the meshes in breadth direction
    B_vec = (B * cosine_spacing(0, 1, num_points)) - (B / 2)

    if symm == 1:
        B_dir = B_vec[int((num_points - 1) / 2):] + org[1]
        n_B = len(B_dir) - 1
        dz = 0.5 * m * th
        n_H = int(np.ceil(H / dz))
        H_dir = - np.linspace(0, H, n_H + 1, endpoint=True) + org[2]
    elif symm == 0:  # TODO: under development
        B_dir = B_vec + org[1]
        n_B = len(B_dir) - 1
        dz = th * m
        n_H = int(np.ceil(H / dz))
        H_dir = - np.linspace(0, H, n_H + 1, endpoint=True) + org[2]

    # Set the body mesh as an instance
    mesh = LO_mesh()
    mesh.origo = org
    mesh.type = 'vertical membrane th'
    if wet_side == 0:
        mesh.wetside = 'exterior'
    else:
        mesh.wetside = 'interior'
    if symm == 0:  # TODO: under development
        mesh.ISX = 0
        mesh.ISY = 0
    else:
        mesh.ISX = 1
        mesh.ISY = 1

    # face
    y_rect, z_rect = np.meshgrid(B_dir, H_dir)
    x_rect = np.zeros_like(y_rect) + (th / 2)

    # bottom
    x_bot, y_bot = np.meshgrid(np.array([0, th / 2]), B_dir)
    z_bot = np.zeros_like(x_bot) - H

    # side
    x_side, z_side = np.meshgrid(np.array([0, th / 2]), H_dir)
    y_side = np.zeros_like(x_side) + (B / 2)

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
    # Membrane face
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
        if wet_side == 1:
            v_2_x = np.append(v_2_x, x_rect[j, 1:n_B + 1])
            v_2_y = np.append(v_2_y, y_rect[j, 1:n_B + 1])
            v_2_z = np.append(v_2_z, z_rect[j, 1:n_B + 1])
            #
            v_4_x = np.append(v_4_x, x_rect[j + 1, :n_B])
            v_4_y = np.append(v_4_y, y_rect[j + 1, :n_B])
            v_4_z = np.append(v_4_z, z_rect[j + 1, :n_B])
        elif wet_side == 0:
            v_2_x = np.append(v_2_x, x_rect[j + 1, :n_B])
            v_2_y = np.append(v_2_y, y_rect[j + 1, :n_B])
            v_2_z = np.append(v_2_z, z_rect[j + 1, :n_B])
            #
            v_4_x = np.append(v_4_x, x_rect[j, 1:n_B + 1])
            v_4_y = np.append(v_4_y, y_rect[j, 1:n_B + 1])
            v_4_z = np.append(v_4_z, z_rect[j, 1:n_B + 1])
        #
    # Membrane side
    for j in range(0, n_H):
        v_1_x = np.append(v_1_x, x_side[j, 0])
        v_1_y = np.append(v_1_y, y_side[j, 0])
        v_1_z = np.append(v_1_z, z_side[j, 0])
        #
        v_3_x = np.append(v_3_x, x_side[j + 1, 1])
        v_3_y = np.append(v_3_y, y_side[j + 1, 1])
        v_3_z = np.append(v_3_z, z_side[j + 1, 1])
        #
        # WAMIT has convention to determine which side is wet and vice versa
        if wet_side == 0:
            v_2_x = np.append(v_2_x, x_side[j, 1])
            v_2_y = np.append(v_2_y, y_side[j, 1])
            v_2_z = np.append(v_2_z, z_side[j, 1])
            #
            v_4_x = np.append(v_4_x, x_side[j + 1, 0])
            v_4_y = np.append(v_4_y, y_side[j + 1, 0])
            v_4_z = np.append(v_4_z, z_side[j + 1, 0])
        elif wet_side == 1:
            v_2_x = np.append(v_2_x, x_side[j + 1, 0])
            v_2_y = np.append(v_2_y, y_side[j + 1, 0])
            v_2_z = np.append(v_2_z, z_side[j + 1, 0])
            #
            v_4_x = np.append(v_4_x, x_side[j, 1])
            v_4_y = np.append(v_4_y, y_side[j, 1])
            v_4_z = np.append(v_4_z, z_side[j, 1])
        #
    # Membrane bottom
    for j in range(0, n_B):
        v_1_x = np.append(v_1_x, x_bot[j, 0])
        v_1_y = np.append(v_1_y, y_bot[j, 0])
        v_1_z = np.append(v_1_z, z_bot[j, 0])
        #
        v_3_x = np.append(v_3_x, x_bot[j + 1, 1])
        v_3_y = np.append(v_3_y, y_bot[j + 1, 1])
        v_3_z = np.append(v_3_z, z_bot[j + 1, 1])
        #
        # WAMIT has convention to determine which side is wet and vice versa
        if wet_side == 0:
            v_2_x = np.append(v_2_x, x_bot[j + 1, 0])
            v_2_y = np.append(v_2_y, y_bot[j + 1, 0])
            v_2_z = np.append(v_2_z, z_bot[j + 1, 0])
            #
            v_4_x = np.append(v_4_x, x_bot[j, 1])
            v_4_y = np.append(v_4_y, y_bot[j, 1])
            v_4_z = np.append(v_4_z, z_bot[j, 1])
        elif wet_side == 1:
            v_2_x = np.append(v_2_x, x_bot[j, 1])
            v_2_y = np.append(v_2_y, y_bot[j, 1])
            v_2_z = np.append(v_2_z, z_bot[j, 1])
            #
            v_4_x = np.append(v_4_x, x_bot[j + 1, 0])
            v_4_y = np.append(v_4_y, y_bot[j + 1, 0])
            v_4_z = np.append(v_4_z, z_bot[j + 1, 0])
    # Array (total panel x 12 coord for vertex 1-4) which it should be okay to use for WAMIT
    mesh.panelmat = np.transpose(np.array([v_1_x, v_1_y, v_1_z,
                                           v_2_x, v_2_y, v_2_z,
                                           v_3_x, v_3_y, v_3_z,
                                           v_4_x, v_4_y, v_4_z]))
    return mesh