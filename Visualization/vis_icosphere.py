import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from icosphere import icosphere

# Parameters
nu = 1 #10  # Subdivision level (controls the resolution)
radius = 1  # Desired radius

# Generate the icosphere
vertices, faces = icosphere(nu)
vertices *= radius  # Scale the vertices

# Compute colors based on Z-values (height)
z_values = vertices[:, 2]
colors = (z_values - z_values.min()) / (z_values.max() - z_values.min())  # Normalize

# Create figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Create face colors using the mean Z-value of each face
face_colors = plt.cm.viridis(colors[faces].mean(axis=1))

# Add 3D mesh
poly = Poly3DCollection(vertices[faces], facecolors="gray", edgecolor='k', alpha=0.8)
ax.add_collection3d(poly)

# Set limits and aspect ratio
ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([-radius, radius])
ax.set_box_aspect([1, 1, 1])


# Hide axis ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

plt.show()
