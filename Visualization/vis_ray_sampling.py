from preprocessing import *
from PCA_resample import *
from find_center_point import *
from graph_cut_funcs import *


path = '.nii'

center_point = np.array([166, 182, 161])

min_dist = 46.25

blob_mask, _ = preprocessing(path)

points = np.argwhere(blob_mask)

num_points = len(points)

downsample_factor = int(num_points / 10000)

points = points[::downsample_factor]



# Define the number of steps and step size
steps = 120
step_size = 1

# Create a Manifold object from the vertices and faces
mesh = create_icosphere(center_point, nu=10, radius=round(min_dist*0.8))

neighbor_list = create_neighbor_list(mesh)

# Initialize empty grid
arr = create_sample_grid_vec_new(mesh, steps, step_size, frac=1)

arr_flat = arr.reshape(-1, 3)

arr_flat = arr_flat[::int(len(arr_flat)/7000)]

interpolator = prepare_nifti_interpolator(path)


values_2d = sample_in_nifti(arr, interpolator, show_plot=True)

cut_mask, maxf = graph_cut_segmentation(values_2d, neighbor_list, delta_x=15, show_plot=False)
print(f"Maximum flow: {maxf}")

new_mask = first_indices_mask(cut_mask)

coordinates, indices = get_mask_coordinates(new_mask, arr)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the main skull point cloud
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c='black', marker='.')
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='r', marker='o', s=100)
ax.scatter(arr_flat[:, 0], arr_flat[:, 1], arr_flat[:, 2], s=5, c='orange', marker='.')

# Optionally, if you want to ensure the axes scale equally:
ax.set_aspect('equal')

# Label axes and set a title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Downsampled Skull Point Cloud with Local Minima')

# Show the combined plot
plt.show()
