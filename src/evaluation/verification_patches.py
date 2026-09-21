"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    verification_patches.py

Description:
    Verification of image and segmentation-mask patches
    generated during dataset preparation.

    The module checks the correspondence between image patches
    and their associated ground-truth masks and helps identify
    potential inconsistencies in the prepared dataset.

Author:
    Hamza Graïn
"""

import cv2
import matplotlib.pyplot as plt

idx = 175

img = cv2.imread(
    f"Dataset_IA/images/biopsie001_patch_{idx:05d}.png"
)

img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

mask = cv2.imread(
    f"Dataset_IA/masks/biopsie001_patch_{idx:05d}.png",
    cv2.IMREAD_GRAYSCALE
)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="jet", vmin=0, vmax=5)
plt.colorbar()
plt.title("Mask")

plt.show()
