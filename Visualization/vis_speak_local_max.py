from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
import nibabel as nib
from skimage.feature import canny
from skimage.color import gray2rgb
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion


# Load and preprocess
path = '.nii'
largest_blob_mask, threshold_value = preprocessing(path)

# Slice through Y-axis at index 201
y_index = 201
binary_slice = largest_blob_mask[:, y_index, :].astype(int)  # shape (X, Z)

# Distance transform (on inverted mask)
distance_map = distance_transform_edt(binary_slice == 0)

# Smoothed distance map
smoothed_distance = gaussian_filter(distance_map, sigma=3)

# Local maxima detection
local_max = peak_local_max(smoothed_distance, min_distance=50, threshold_abs=0.1, exclude_border=True)

# Remove any out-of-bounds coordinates just in case
xmax, zmax = binary_slice.shape
valid = (local_max[:, 0] < xmax) & (local_max[:, 1] < zmax)
local_max = local_max[valid]

# Extract white edges from binary mask
edges = canny(binary_slice.astype(bool), sigma=3.5)
edges_T = edges.T  # Transpose to match plot orientation

# Prepare RGB images with white edges overlaid
def overlay_edges_gray_background(gray_image, edge_mask_T):
    """Overlay white edges on a grayscale image."""
    rgb = np.stack([gray_image]*3, axis=-1)  # shape (..., ..., 3)
    rgb /= rgb.max() if rgb.max() > 0 else 1  # Normalize to [0,1]
    rgb[edge_mask_T, :] = [1, 1, 1]  # Overlay white edges
    return rgb

# Create overlay images (only for plots 1 to 3)
images_rgb = [
    binary_slice.T,  # Raw binary mask (no overlay here)
    overlay_edges_gray_background(distance_map.T, edges_T),
    overlay_edges_gray_background(smoothed_distance.T, edges_T),
    overlay_edges_gray_background(smoothed_distance.T, edges_T),
]

# Titles
titles = ["Binary Mask", "Distance Transformed", "Smoothed Distance Transformed", "Local Maxima Points"]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.imshow(images_rgb[i], cmap='gray', origin='lower')  # No overlay on binary mask
    else:
        ax.imshow(images_rgb[i], origin='lower')  # RGB image with white edges
    if i == 3:
        ax.scatter(local_max[:, 0], local_max[:, 1], c='red', s=15)  # Red dots for maxima
    ax.set_title(titles[i], fontsize=12)
    ax.axis('off')

plt.tight_layout(pad=0.3)
plt.show()
