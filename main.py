import numpy as np
import os
import mesh

tes = mesh.hollow_cylinder(1.5, 1.5, 1.5, 5, 5, 5, 1, 0)
tes_rosetta = mesh.to_rosetta_format(tes)
mesh.gdf_LO(tes, 'C://Users//muhamm//programming//WAMIT_files//test_cases//', 'check', 'check', 1, 1)