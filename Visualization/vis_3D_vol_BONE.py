import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
import pyvista as pv

# Load NIfTI volume
path = '...'
img = nib.load(path)
data = img.get_fdata()

# Thresholding for bone visualization
thresh = threshold_otsu(data)
opacity = [0.001 if v < thresh else 0.4 for v in np.linspace(np.min(data), np.max(data), 30)]

# Create ImageData for the volume
grid = pv.ImageData()
grid.dimensions = np.array(data.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)  # Update if your image has real spacing
grid.cell_data["values"] = data.flatten(order="F")

# Create a bounding box (outer shell lines)
shape = np.array(data.shape)
box = pv.Box(bounds=(0, shape[0], 0, shape[1], 0, shape[2]))

# Set up the plotter
plotter = pv.Plotter()
plotter.add_volume(grid, cmap="gray", opacity=opacity)
plotter.add_mesh(box, color='black', style='wireframe', line_width=2)

# Show the rendering
plotter.show()
