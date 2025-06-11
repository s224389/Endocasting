from find_center_point import *
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


path = '.nii'
nii_vol = nib.load(path)
xdim, ydim, zdim = nii_vol.shape


local_max, local_max_values, points = find_center_points(path)

point, value = select_local_max(points, local_max, local_max_values)


P = point

# Compute half-sizes
dx = 0.6 * xdim *0.6
dy = ydim *0.6
dz = zdim *0.6

# Define 8 corners of the box centered at P
corners = np.array([
    [P[0] - dx, P[1] - dy, P[2] - dz],
    [P[0] + dx, P[1] - dy, P[2] - dz],
    [P[0] + dx, P[1] + dy, P[2] - dz],
    [P[0] - dx, P[1] + dy, P[2] - dz],
    [P[0] - dx, P[1] - dy, P[2] + dz],
    [P[0] + dx, P[1] - dy, P[2] + dz],
    [P[0] + dx, P[1] + dy, P[2] + dz],
    [P[0] - dx, P[1] + dy, P[2] + dz]
])

# Define edges (pairs of corner indices)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
]

# Create lines for edges
lines = [(corners[start], corners[end]) for start, end in edges]

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="black", marker="o")
ax.scatter(point[0], point[1], point[2], s=300, c="red", marker="o")

# Add box edges
line_collection = Line3DCollection(lines, colors='red', linewidths=1.5)
ax.add_collection3d(line_collection)

ax.set_aspect('equal')
plt.show()
