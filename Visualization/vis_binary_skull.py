from preprocessing import *

path = '.nii'

blob_mask, _ = preprocessing(path)

points = np.argwhere(blob_mask)

num_points = len(points)

downsample_factor = int(num_points / 10000)

points = points[::downsample_factor]


# PLOT
# Plot the point cloud and local minima using Matplotlib's 3D plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Option A: treat [0] as x, [1] as y, [2] as z for both sets
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="black", marker="o")
ax.set_aspect('equal')


plt.show()
