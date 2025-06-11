import sys
import os
import numpy as np

# Now import functions from your script
from graph_cut_funcs import *

steps0 = 220
step_size0 = 0.5
taubin_iter = 10
center_point = np.array([212, 188, 164])
manifold = create_icosphere(center_point,nu=6, radius=20)
neighbor_list = create_neighbor_list(manifold)
samples = create_sample_grid_vec_new(manifold, steps0, step_size0, frac=1)

path = ".nii"
interpolator = prepare_nifti_interpolator(path)
values_2d = sample_in_nifti(samples, interpolator, show_plot=False)
plt.imshow(values_2d, cmap='gray', origin='lower')
plt.show()
