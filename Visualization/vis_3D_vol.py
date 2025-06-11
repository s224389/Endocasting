# Creates a 3D rendering of the nifti volume (with emphasis on the bone)
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
import pyvista as pv

path = '.nii'

img = nib.load(path)
data = img.get_fdata()

thresh = threshold_otsu(data)
opacity = [0.01 if v < thresh else 0.1 for v in np.linspace(np.min(data), np.max(data), 30)]

# Use ImageData instead of UniformGrid
grid = pv.ImageData()
grid.dimensions = np.array(data.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = data.flatten(order="F")

# Set up the plotter
plotter = pv.Plotter()
plotter.add_volume(grid, cmap="gray", opacity=opacity) #clim=[0, 100]
plotter.show()
