"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    predict_wsi.py

Description:
    Whole Slide Image inference using the trained semantic
    segmentation model.

    The script processes the WSI using overlapping patches,
    generates class probability maps, and reconstructs the
    predicted segmentation mask at slide level.

Author:
    Hamza Graïn
"""

import openslide
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm


# CONFIGURATION


NDPI_PATH = r"path\to\the\biopsie.ndpi"

MODEL_PATH = r"path\to\the\best_model_unet.pth"

LEVEL = 4

PATCH_SIZE = 512
STRIDE = 256

NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# COULEURS
# =====================================================

COLORS = {
    0: (0, 0, 0),          # Fond
    1: (0, 255, 255),      # CK14 (cyan)
    2: (255, 165, 0),      # Stroma (orange)
    3: (128, 128, 128),    # Autres (gris)
}

# =====================================================
# MODELE
# =====================================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# =====================================================
# OUVERTURE WSI
# =====================================================

slide = openslide.OpenSlide(NDPI_PATH)

width, height = slide.level_dimensions[LEVEL]

downsample = slide.level_downsamples[LEVEL]

print(f"WSI : {width} x {height}")
print(f"Downsample : {downsample}")

# =====================================================
# ACCUMULATION
# =====================================================

scores = np.zeros(
    (NUM_CLASSES, height, width),
    dtype=np.float32
)

counter = np.zeros(
    (height, width),
    dtype=np.float32
)

# =====================================================
# PREDICTION
# =====================================================

ys = list(range(0, height, STRIDE))
xs = list(range(0, width, STRIDE))

for y in tqdm(ys):

    for x in xs:

        patch_w = min(PATCH_SIZE, width - x)
        patch_h = min(PATCH_SIZE, height - y)

        region = slide.read_region(
            (
                int(x * downsample),
                int(y * downsample)
            ),
            LEVEL,
            (patch_w, patch_h)
        )

        patch = np.array(region)[:, :, :3]

        # ignorer uniquement le vrai fond

        if patch.mean() > 245:
            continue

        image = patch.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        image = image.to(DEVICE)

        with torch.no_grad():

            pred = model(image)

            pred = torch.softmax(
                pred,
                dim=1
            )

        pred = pred.cpu().numpy()[0]

        scores[
            :,
            y:y+patch_h,
            x:x+patch_w
        ] += pred[:, :patch_h, :patch_w]

        counter[
            y:y+patch_h,
            x:x+patch_w
        ] += 1

# =====================================================
# MOYENNE
# =====================================================

counter[counter == 0] = 1

scores /= counter[np.newaxis, :, :]

prediction_map = np.argmax(
    scores,
    axis=0
).astype(np.uint8)

print("\nClasses présentes :")
print(np.unique(prediction_map))

# =====================================================
# VISUALISATION
# =====================================================

overlay = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

for cls, color in COLORS.items():

    overlay[
        prediction_map == cls
    ] = color

# =====================================================
# SAUVEGARDE
# =====================================================

cv2.imwrite(
    "prediction_mask.png",
    prediction_map
)

cv2.imwrite(
    "prediction_overlay.png",
    cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )
)

print("\n================================")
print("Prediction terminée")
print("prediction_mask.png")
print("prediction_overlay.png")
print("================================")
