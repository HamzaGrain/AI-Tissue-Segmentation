"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    inference.py

Description:
    Whole Slide Image inference using the trained semantic
    segmentation model.

    The module processes the slide using overlapping patches,
    accumulates class probabilities, and reconstructs the
    final segmentation prediction and confidence map.

Author:
    Hamza Graïn
"""

import torch
import numpy as np
from tqdm import tqdm

from config import *


# INFERENCE


def predict_wsi(model, slide_info):

    slide = slide_info["slide"]
    width = slide_info["width"]
    height = slide_info["height"]
    downsample = slide_info["downsample"]

    print("=" * 60)
    print("Début de la prédiction")
    print("=" * 60)

    # ------------------------------------------------------
    # Accumulation des probabilités
    # ------------------------------------------------------

    scores = np.zeros(
        (NUM_CLASSES, height, width),
        dtype=np.float32
    )

    counter = np.zeros(
        (height, width),
        dtype=np.float32
    )

    ys = list(range(0, height, STRIDE))
    xs = list(range(0, width, STRIDE))

    with torch.no_grad():

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
                    (
                        patch_w,
                        patch_h
                    )
                )

                patch = np.array(region)[:, :, :3]

                # Ignorer uniquement le fond
                if patch.mean() > 245:
                    continue

                image = patch.astype(np.float32) / 255.0

                image = np.transpose(
                    image,
                    (2, 0, 1)
                )

                image = torch.tensor(
                    image,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                logits = model(image)

                probs = torch.softmax(
                    logits,
                    dim=1
                )

                probs = probs.cpu().numpy()[0]

               
                # Fusion IDENTIQUE à predict_wsi.py
                

                scores[
                    :,
                    y:y + patch_h,
                    x:x + patch_w
                ] += probs[:, :patch_h, :patch_w]

                counter[
                    y:y + patch_h,
                    x:x + patch_w
                ] += 1

    # ------------------------------------------------------
    # Moyenne
    # ------------------------------------------------------

    counter[counter == 0] = 1

    scores /= counter[np.newaxis, :, :]

    # ------------------------------------------------------
    # Résultat final
    # ------------------------------------------------------

    prediction = np.argmax(
        scores,
        axis=0
    ).astype(np.uint8)

    confidence = np.max(
        scores,
        axis=0
    )

    print()
    print("Prédiction terminée.")
    print()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": scores
    }
