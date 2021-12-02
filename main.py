import numpy as np
import os
import mesh
import input_files

# Create mesh instance
with mesh.Timer('Mesh'):
    tes = mesh.vertical_membrane_cos(0.595, 1, 10, 0.595 / 4, 1, 0)
    # tes = mesh.rectangular(0.595, 1, 10, 100, 1, 0, org=None)
    # tes = mesh.hollow_cylinder(0, 1.5, 1.5, 800, 800, 800, 1, 0)

tes_rosetta = mesh.to_rosetta_format(tes.panelmat)
input_files.gdf_LO(tes.panelmat, 'C://Users//muhamm//programming//WAMIT_files//test_cases//', 'check', 'check', tes.ISX, tes.ISY)

input_files.pot_ctrlfile('C://Users//muhamm//programming//WAMIT_files//test_cases//', 'check', 'check', np.array([1,1,0,0,0,0,0,0]), np.array([0.1,0.2,0.3]), np.array([]), HBOT=-1)