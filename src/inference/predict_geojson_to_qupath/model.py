"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    model.py

Description:
    Model and Whole Slide Image management for the
    automated segmentation pipeline.

    The module handles model loading, slide loading,
    and the preparation of information required for inference.

Author:
    Hamza Graïn
"""

import torch
import openslide
import segmentation_models_pytorch as smp

from config import *


# CHARGEMENT DU MODELE


def load_model():

    print("=" * 60)
    print("Chargement du modèle...")
    print("=" * 60)

    model = smp.Unet(

        encoder_name=ENCODER_NAME,

        encoder_weights=ENCODER_WEIGHTS,

        in_channels=IN_CHANNELS,

        classes=NUM_CLASSES

    )

    state_dict = torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    print("Modèle chargé.")

    if DEVICE == "cuda":

        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    else:

        print("CPU")

    print()

    return model



# CHARGEMENT DE LA WSI


def load_slide():

    print("=" * 60)
    print("Ouverture de la lame...")
    print("=" * 60)

    slide = openslide.OpenSlide(
        NDPI_PATH
    )

    base_width, base_height = slide.level_dimensions[0]

    width, height = slide.level_dimensions[LEVEL]

    downsample = slide.level_downsamples[LEVEL]

    info = {

        "slide": slide,

        "width": width,

        "height": height,

        "base_width": base_width,

        "base_height": base_height,

        "downsample": downsample

    }

    print()

    print(
        f"Niveau 0 : {base_width} x {base_height}"
    )

    print(
        f"Niveau {LEVEL} : {width} x {height}"
    )

    print(
        f"Downsample : {downsample:.2f}"
    )

    print()

    return info



# INITIALISATION DES CARTES


def initialize_prediction_maps(width, height):

    scores = torch.zeros(

        (
            NUM_CLASSES,
            height,
            width
        ),

        dtype=torch.float32

    )

    counter = torch.zeros(

        (
            height,
            width
        ),

        dtype=torch.float32

    )

    return scores, counter



# RESUME


def print_summary(info):

    print("=" * 60)

    print("Résumé")

    print("=" * 60)

    print(f"WSI          : {NDPI_PATH}")

    print(f"Model        : {MODEL_PATH}")

    print(f"Encoder      : {ENCODER_NAME}")

    print(f"Classes      : {NUM_CLASSES}")

    print(f"Patch size   : {PATCH_SIZE}")

    print(f"Stride       : {STRIDE}")

    print(f"Device       : {DEVICE}")

    print(f"Level        : {LEVEL}")

    print(f"Dimensions   : {info['width']} x {info['height']}")

    print("=" * 60)

    print()
