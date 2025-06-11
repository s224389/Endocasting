import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# Load the image and convert to grayscale
img = Image.open('.png').convert('L')
img_np = np.array(img)

diff_img = np.zeros((8,10))

for j in range(0, img_np.shape[1]):
    for i in range(0, img_np.shape[0]-1):
        diff_img[i,j] = img_np[i+1,j] - img_np[i,j]


# Plot the difference image
plt.imshow(diff_img, cmap='gray')
plt.colorbar()
plt.title('Difference Image')
plt.show()
