"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    generate_patches.py

Description:
    Generation of image and segmentation-mask patches from
    whole-slide images and their corresponding ground-truth masks.

    The generated patches are used to prepare the dataset
    for training and evaluating the segmentation models.

Author:
    Hamza Graïn
"""

import os
import cv2
import numpy as np
import openslide
from tqdm import tqdm


# CONFIGURATION


# Select the biopsy and its corresponding segmentation mask.
# The biopsy name, slide and mask must refer to the same biopsy.
# Example: BIOPSY_NAME = "biopsie114"
#           NDPI_PATH  -> Biopsie_114.ndpi
#           MASK_PATH  -> mask_114.png

BIOPSY_NAME = "biopsie_number"

NDPI_PATH = r"path\to\your\slide.ndpi"
MASK_PATH = r"path\to\your\mask.png"

LEVEL = 4

PATCH_SIZE = 512
STRIDE = 256

# Directory containing the input image patches
IMAGE_DIR = "Dataset_IA/images"

# Directory containing the corresponding ground-truth masks
MASK_DIR = "Dataset_IA/masks"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# Seuil minimal de pixels annotés (hors fond)
MIN_TISSUE_RATIO = 0.05  # 5 %

# ======================================================
# CHARGEMENT
# ======================================================

slide = openslide.OpenSlide(NDPI_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Impossible de charger : {MASK_PATH}")

height, width = mask.shape

print(f"Masque : {width} x {height}")

downsample = slide.level_downsamples[LEVEL]

patch_id = 0

# ======================================================
# EXTRACTION DES PATCHS
# ======================================================

for y in tqdm(range(0, height - PATCH_SIZE + 1, STRIDE), desc="Extraction"):

    for x in range(0, width - PATCH_SIZE + 1, STRIDE):

        mask_patch = mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        # Ignorer les patchs entièrement vides
        if np.all(mask_patch == 0):
            continue

        # Calcul de la proportion de pixels annotés
        tissue_ratio = np.mean(mask_patch != 0)

        # Ignorer les patchs contenant trop peu de tissu
        if tissue_ratio < MIN_TISSUE_RATIO:
            continue

        region = slide.read_region(
            (
                int(x * downsample),
                int(y * downsample)
            ),
            LEVEL,
            (PATCH_SIZE, PATCH_SIZE)
        )

        image_patch = np.array(region)[:, :, :3]
        image_patch = cv2.cvtColor(
            image_patch,
            cv2.COLOR_RGB2BGR
        )

        filename = f"{BIOPSY_NAME}_patch_{patch_id:05d}.png"

        cv2.imwrite(
            os.path.join(IMAGE_DIR, filename),
            image_patch
        )

        cv2.imwrite(
            os.path.join(MASK_DIR, filename),
            mask_patch
        )

        patch_id += 1

print("\n===================================")
print(f"Biopsie : {BIOPSY_NAME}")
print(f"Patches sauvegardés : {patch_id}")
print("===================================")
