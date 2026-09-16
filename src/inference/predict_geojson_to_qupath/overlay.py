"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    overlay.py

Description:
    Visualization of segmentation predictions and
    confidence information.

    The module generates color masks, image overlays,
    segmentation contours, confidence maps, and legends
    for visual inspection of the predictions.

Author:
    Hamza Graïn
"""

import os
import cv2
import numpy as np

from config import *


# COULEURS


def create_color_mask(mask):

    h, w = mask.shape

    color = np.zeros(

        (h, w, 3),

        dtype=np.uint8

    )

    for cls, rgb in CLASS_COLORS.items():

        color[
            mask == cls
        ] = rgb

    return color



# OVERLAY


def create_overlay(image, mask_color, alpha=0.45):

    overlay = cv2.addWeighted(

        image,

        1-alpha,

        mask_color,

        alpha,

        0

    )

    return overlay



# CARTE DE CONFIANCE

def create_confidence_map(confidence):

    confidence = np.clip(
        confidence,
        0,
        1
    )

    confidence = (
        confidence * 255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(

        confidence,

        cv2.COLORMAP_JET

    )

    return heatmap



# CONTOURS


def draw_contours(image, mask):

    output = image.copy()

    for cls in CLASS_NAMES:

        if cls == 0:
            continue

        binary = (

            mask == cls

        ).astype(np.uint8)

        contours, _ = cv2.findContours(

            binary,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )

        cv2.drawContours(

            output,

            contours,

            -1,

            CLASS_COLORS[cls],

            2

        )

    return output



# LEGENDE


def create_legend():

    legend = np.ones(

        (180, 320, 3),

        dtype=np.uint8

    ) * 255

    y = 30

    for cls in CLASS_NAMES:

        if cls == 0:
            continue

        cv2.rectangle(

            legend,

            (20, y-12),

            (50, y+12),

            CLASS_COLORS[cls],

            -1

        )

        cv2.putText(

            legend,

            CLASS_NAMES[cls],

            (70, y+7),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0,0,0),

            2

        )

        y += 45

    return legend


# ==========================================================
# EXPORT
# ==========================================================

def export_visualizations(

    slide_info,

    prediction,

    clean_prediction,

    confidence

):

    print("="*60)
    print("Création des visualisations")
    print("="*60)

    slide = slide_info["slide"]

    width = slide_info["width"]

    height = slide_info["height"]

    image = np.array(

        slide.read_region(

            (0,0),

            LEVEL,

            (width,height)

        )

    )[:,:,:3]

    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True

    )

    mask_color = create_color_mask(

        clean_prediction

    )

    overlay = create_overlay(

        image,

        mask_color

    )

    contours = draw_contours(

        image,

        clean_prediction

    )

    heatmap = create_confidence_map(

        confidence

    )

    legend = create_legend()

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            MASK_NAME

        ),

        prediction

    )

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            MASK_CLEAN_NAME

        ),

        clean_prediction

    )

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            OVERLAY_NAME

        ),

        cv2.cvtColor(

            overlay,

            cv2.COLOR_RGB2BGR

        )

    )

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            "prediction_contours.png"

        ),

        cv2.cvtColor(

            contours,

            cv2.COLOR_RGB2BGR

        )

    )

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            "confidence_map.png"

        ),

        heatmap

    )

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            "legend.png"

        ),

        cv2.cvtColor(

            legend,

            cv2.COLOR_RGB2BGR

        )

    )

    print()

    print("Visualisations sauvegardées.")

    print()

    print(OUTPUT_DIR)
