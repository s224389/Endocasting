from preprocessing import *
from PCA_resample import *
from find_center_point import *

########### NOTE THAT: find_center_point.py has been tweaked for this to make the visualization as wanted! ###########
# This is NOT in the repository.

path = '.nii'

# Load NifTi file and resample i PCA-space
_ = PCA_resample_GUI(path)

pca_path = path.replace('.nii', '_PCA_resampled.nii')

local_max, local_max_values, points = find_center_points(pca_path)

print(f'Number of local maxima: {len(local_max)}')

point, value = select_local_max(points, local_max, local_max_values)

print(f'Selected point: {point}')
print(f'Value at selected point: {value}')



# PLOT
# Plot the point cloud and local minima using Matplotlib's 3D plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Option A: treat [0] as x, [1] as y, [2] as z for both sets
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="black", marker="o")
ax.scatter(point[0], point[1], point[2], s=300, c="red", marker="o")
ax.set_aspect('equal')
plt.show()
