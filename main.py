
# ==============================================================================
# Digital Image Processing (DIP)
# Wavelet Domain Filtering:
# 1. Ideal Low Pass & Ideal High Pass Filters
# 2. Gaussian Low Pass & Gaussian High Pass Filters
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt

# Read image and convert to grayscale
img = Image.open("image (2).png").convert("L")
img_array = np.array(img, dtype=np.float32)


# ------------------------------------------------------------------------------
# 1. Wavelet Decomposition
# ------------------------------------------------------------------------------

wavelet = 'haar'

LL, (LH, HL, HH) = pywt.dwt2(img_array, wavelet)

# LL: Low frequency
# LH, HL, HH: High frequency components


# ------------------------------------------------------------------------------
# 2. Ideal Low Pass & Ideal High Pass Filters
# ------------------------------------------------------------------------------

rows, cols = LL.shape
crow, ccol = rows // 2, cols // 2

R = 50

# Create Ideal Low Pass Mask
mask_Low = np.zeros((rows, cols), dtype=np.float32)

for u in range(rows):
    for v in range(cols):
        D = np.sqrt((u - crow)**2 + (v - ccol)**2)
        if D <= R:
            mask_Low[u, v] = 1

# Create Ideal High Pass Mask
mask_High = 1 - mask_Low

# Apply Ideal Low Pass
LL_low = LL * mask_Low
LH_low = LH * mask_Low
HL_low = HL * mask_Low
HH_low = HH * mask_Low

# Apply Ideal High Pass
LL_high = LL * mask_High
LH_high = LH * mask_High
HL_high = HL * mask_High
HH_high = HH * mask_High

# Reconstruct images
img_Ideal_Low = pywt.idwt2(
    (LL_low, (LH_low, HL_low, HH_low)), wavelet
)

img_Ideal_High = pywt.idwt2(
    (LL_high, (LH_high, HL_high, HH_high)), wavelet
)


# ------------------------------------------------------------------------------
# 3. Gaussian Low Pass & Gaussian High Pass Filters
# ------------------------------------------------------------------------------

mask_G_low = np.zeros((rows, cols), dtype=np.float32)

for u in range(rows):
    for v in range(cols):
        D = np.sqrt((u - crow)**2 + (v - ccol)**2)
        mask_G_low[u, v] = np.exp(-(D**2) / (2 * R**2))

# Gaussian High Pass Mask
mask_G_high = 1 - mask_G_low

# Apply Gaussian Low Pass
LL_G_low = LL * mask_G_low
LH_G_low = LH * mask_G_low
HL_G_low = HL * mask_G_low
HH_G_low = HH * mask_G_low

# Apply Gaussian High Pass
LL_G_high = LL * mask_G_high
LH_G_high = LH * mask_G_high
HL_G_high = HL * mask_G_high
HH_G_high = HH * mask_G_high

# Reconstruct images
img_Gaussian_Low = pywt.idwt2(
    (LL_G_low, (LH_G_low, HL_G_low, HH_G_low)), wavelet
)

img_Gaussian_High = pywt.idwt2(
    (LL_G_high, (LH_G_high, HL_G_high, HH_G_high)), wavelet
)


# ------------------------------------------------------------------------------
# 4. Display Results
# ------------------------------------------------------------------------------

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img_array, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Ideal Low Pass - Wavelet")
plt.imshow(img_Ideal_Low, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Ideal High Pass - Wavelet")
plt.imshow(img_Ideal_High, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Gaussian Low Pass - Wavelet")
plt.imshow(img_Gaussian_Low, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Gaussian High Pass - Wavelet")
plt.imshow(img_Gaussian_High, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
