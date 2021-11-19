import os
import mesh.low_order_lib


def gdf_LO(panel_mat, folder, gdf_filename, desc, ISX, ISY, g=9.81, l=1.00):
    """
    SHORT DESCRIPTION:
    Function to generate name_id.gdf file according to WAMIT v7.3 manual
    for LOW-ORDER panel method.
    INPUT:
    - panel_mat: matrix output from low_order_lib.py
    - folder: location of the folder
    - gdf_file: the desired name of the .gdf file
    - desc: short description of the .gdf file
    - g: gravity acceleration
    - l: characteristic length of the structure
    - ISX: 0 = x=0 is not geometry symmetry, 1 = x=0 is geometry symmetry
    - ISY: 0 = y=0 is not geometry symmetry, 1 = y=0 is geometry symmetry
      if ISX or ISY = -1 or -2, hence x=0 and y=0 is a plane wall
      respectively

    OUTPUT:
    -folder/gdf_filename.gdf
    """
    n_panel = int(panel_mat.shape[0])
    panel_rosetta = mesh.to_rosetta_format(panel_mat)

    if not os.path.exists(folder):
        os.makedirs(folder)
    gdf_file = open(folder + "/{}.gdf".format(gdf_filename), "w")
    gdf_file.write('{}.gdf'.format(gdf_filename).upper() + ' -- ' + '{}'.format(desc))
    gdf_file.write('\n')
    gdf_file.write('    '+'{:5.3f}'.format(l)+'      '+'{:5.3f}'.format(g) +
                   '      '+'ULEN, GRAV')
    gdf_file.write('\n')
    gdf_file.write('       ' +'{:>2}'.format(ISX) + '         ' +
                   '{:>2}'.format(ISY) +
                   '      ' + 'ISX, ISY')

    gdf_file.write('\n')
    gdf_file.write('    '+'{:5}'.format(n_panel))
    gdf_file.write('                 '+'NPAN')
    gdf_file.write('\n')

    panel_string = '   ' +\
                   ('\n' + '   ').join('     '.join('{:12.7f}'.format(x) for x in y) for y in panel_rosetta)
    gdf_file.write(panel_string)
    gdf_file.close()
    return gdf_file

