
# -----------------------------------------------------------
# Bimodal-histogram visualisation + Otsu threshold for a μCT
# -----------------------------------------------------------

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu        # scikit-image

# 1. Load the volume ----------------------------------------------------------
nii_path = '.nii'             # change to your file
img       = nib.load(nii_path)
data      = img.get_fdata(dtype=np.float32)      # cast once to save RAM

# 2. Flatten & (optionally) clean --------------------------------------------
voxels = data.ravel()

# If the scan has a padded background (e.g. −1024 HU or exactly 0), 
# you can drop it to keep the histogram readable.  Example:
# voxels = voxels[voxels > -500]                 # tweak to suit your data

# 3. Compute Otsu threshold ----------------------------------------------------
t_otsu = threshold_otsu(voxels)

# 4. Plot ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(
    voxels,
    bins=512,                # enough bins to show two peaks
    density=False,            # Counts
    alpha=0.7,
    label="Voxel intensities"
)
plt.axvline(
    t_otsu,
    linestyle="--",
    linewidth=2,
    label=f"Otsu threshold = {t_otsu:0.1f}",
    color='black'
)
plt.title("Histogram of μCT intensities")
plt.xlabel("Raw intensity")
plt.ylabel("Count")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
