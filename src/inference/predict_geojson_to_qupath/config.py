"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    config.py

Description:
    Centralized configuration for the automated tissue
    segmentation and QuPath annotation export pipeline.

    The module defines input and output paths, model parameters,
    post-processing settings, quality assurance thresholds,
    tissue classes, visualization colors, and export options.

Author:
    Hamza Graïn
"""

import os
import torch

# ==========================================================
# PATHS
# ==========================================================

# Whole Slide Image (.ndpi)

NDPI_PATH = (
    r"path\to\the\biopsie.ndpi"
)

# Modèle entraîné

MODEL_PATH = (
    r"path\to\the\best_model_unet.pth"
)

# Dossier de sortie

OUTPUT_DIR = "results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================================
# OPENSLIDE
# ==========================================================

# Niveau de lecture
# 0 = résolution maximale

LEVEL = 4

# ==========================================================
# PATCHES
# ==========================================================

PATCH_SIZE = 512

STRIDE = 256

# ==========================================================
# MODELE
# ==========================================================

ENCODER_NAME = "resnet34"

ENCODER_WEIGHTS = None

NUM_CLASSES = 4

IN_CHANNELS = 3

# ==========================================================
# GPU
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# POST-PROCESSING
# ==========================================================

MIN_CONFIDENCE = 0.65

MIN_REGION_AREA = 200

SMOOTH_KERNEL = 3

SIMPLIFICATION = 0

MERGE_DISTANCE = 0

# ==========================================================
# EXPORT
# ==========================================================

EXPORT_MASK = True

EXPORT_OVERLAY = True

EXPORT_GEOJSON = True

EXPORT_STATISTICS = True

EXPORT_REPORT = True

EXPORT_WARNINGS = True

EXPORT_CSV = True

# ==========================================================
# QUALITY ASSURANCE
# ==========================================================

QA_MIN_CONFIDENCE = 0.75

QA_SMALL_REGION = 1500

QA_THIN_REGION = 10

# ==========================================================
# CLASSES
# ==========================================================

CLASS_NAMES = {

    0: "Fond",

    1: "CK14",

    2: "Stroma",

    3: "Autres"

}

# ==========================================================
# COULEURS QUPATH
# ==========================================================

# RGB
# Les noms doivent être identiques à ceux présents dans QuPath

CLASS_COLORS = {

    0: (0, 0, 0),

    1: (0, 255, 255),      # CK14

    2: (255, 165, 0),      # Stroma

    3: (102, 26, 51)       # Autres (couleur exportée par QuPath)

}

# ==========================================================
# COULEURS QA
# ==========================================================

QA_COLORS = {

    "GOOD": (0, 255, 0),

    "MEDIUM": (255, 165, 0),

    "LOW": (255, 0, 0)

}

# ==========================================================
# NOMS DES FICHIERS
# ==========================================================

MASK_NAME = "prediction_mask.png"

MASK_CLEAN_NAME = "prediction_clean.png"

OVERLAY_NAME = "prediction_overlay.png"

GEOJSON_NAME = "prediction.geojson"

STATISTICS_NAME = "annotation_statistics.csv"

REPORT_NAME = "annotation_report.txt"

WARNINGS_NAME = "annotation_warnings.csv"

# ==========================================================
# AFFICHAGE
# ==========================================================

VERBOSE = True

SHOW_PROGRESS = True

PRINT_CLASSES = True

SAVE_INTERMEDIATE = True

# ==========================================================
# VERSION
# ==========================================================

PROJECT_NAME = "Predict Annotation"

VERSION = "2.1"
