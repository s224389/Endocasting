from preprocessing import *
from PCA_resample import *
from find_center_point import *
from graph_cut_funcs import *

# By Christian Bjerrgaard and Aske Rove
# For our bachelor thesis at The Technical University of Denmark
# Last edited: 10-06-2025


########### FILL OUT ###########
path = ".nii"  # Replace with your actual NifTi file path
out_path = ".../"  # Replace with your desired output path
num_iters = [10,5,3,2]
step_sizes = [1,1,0.5,0.5]
################################



# Load NifTi file and resample in PCA-space
_ = PCA_resample_GUI(path)

pca_path = path.replace('.nii', '_PCA_resampled.nii')

local_max, local_max_values, points = find_center_points(pca_path)
print(f'Number of local maxima: {len(local_max)}')


center_point, min_dist = select_local_max(points, local_max, local_max_values)
print(f'Selected center point: {center_point}')
print(f'Minimum distance value at selected center point: {min_dist}')

#iterative_mesh(center_point, min_dist, out_path, pca_path) #outputs the first iterative mesh and last
iterative_mesh_every_iter(center_point, min_dist, out_path, pca_path,num_iters=num_iters,step_sizes=step_sizes) #outputs all iterative meshes