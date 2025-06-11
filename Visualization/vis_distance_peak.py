from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
import nibabel as nib

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

# Prepare images (transposed for correct visual orientation)
images = [binary_slice.T, distance_map.T, smoothed_distance.T, smoothed_distance.T]
titles = ["Binary Mask", "Distance Transformed", "Smoothed Distance Transformed", "Local Maxima Points"]

# Plotting in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray', origin='lower')
    if i == 3:
        # Overlay red dots (x=z, y=x after transpose)
        ax.scatter(local_max[:, 0], local_max[:, 1], c='red', s=15)
    ax.set_title(titles[i], fontsize=12)
    ax.axis('off')

plt.tight_layout(pad=0.3)
plt.show()
