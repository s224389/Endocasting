# Creates a 3D rendering of the nifti volume (with emphasis on the bone)
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
import pyvista as pv
from icosphere import icosphere


path = '...'

img = nib.load(path)
data = img.get_fdata()

thresh = threshold_otsu(data)
opacity = [0.015 if v < thresh else 0.1 for v in np.linspace(np.min(data), np.max(data), 30)]

# Use ImageData instead of UniformGrid
grid = pv.ImageData()
grid.dimensions = np.array(data.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = data.flatten(order="F")

# Set up the plotter
plotter = pv.Plotter()
plotter.add_volume(grid, cmap="gray", opacity=opacity) #clim=[0, 100]



 # Generate the icosphere
vertices, faces = icosphere(10)
vertices *= 35
# Center icosphere at the specified point
vertices = vertices + np.array([[169, 186, 160]])
# Ensure vertices and faces are NumPy arrays with correct data types
vertices = np.asarray(vertices, dtype=np.float64)
faces = np.asarray(faces, dtype=np.int32)
vtk_faces = np.hstack([[3, *face] for face in faces])
mesh = pv.PolyData(vertices, vtk_faces)


brain_mesh  = pv.read('...')
plotter.add_mesh(mesh,     color='red',   opacity=1)


plotter.show()
