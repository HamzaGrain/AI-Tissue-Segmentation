"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    polygon_extraction.py

Description:
    Extraction of polygon geometries from segmentation masks.

    The module identifies the contours of each tissue class,
    converts them into valid polygon geometries, and prepares
    them for subsequent geometric processing and export.

Author:
    Hamza Graïn
"""

import cv2
import numpy as np

from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

from config import *


# Extraction des polygones d'une classe


def extract_polygons(mask, class_id):

    binary = (mask == class_id).astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE
    )

    polygons = []

    if hierarchy is None:
        return polygons

    hierarchy = hierarchy[0]

    # contour externe
    for idx, cnt in enumerate(contours):

        parent = hierarchy[idx][3]

        if parent != -1:
            continue

        if len(cnt) < 4:
            continue

        exterior = cnt[:, 0, :]

        holes = []

        child = hierarchy[idx][2]

        while child != -1:

            hole = contours[child][:, 0, :]

            if len(hole) >= 4:
                holes.append(hole)

            child = hierarchy[child][0]

        try:

            poly = Polygon(exterior, holes)

            # Corrige automatiquement les géométries invalides
            if not poly.is_valid:
                poly = make_valid(poly)

            if poly.is_empty:
                continue

            if isinstance(poly, MultiPolygon):
                polygons.extend(list(poly.geoms))
            else:
                polygons.append(poly)

        except Exception:
            continue

    return polygons



# Extraction de toutes les classes


def extract_all_polygons(mask):

    print("=" * 60)
    print("Extraction des polygones")
    print("=" * 60)

    annotations = []

    for class_id in range(1, NUM_CLASSES):

        polys = extract_polygons(mask, class_id)

        print(f"{CLASS_NAMES[class_id]} : {len(polys)} polygones")

        for poly in polys:

            annotations.append({

                "polygon": poly,

                "class_id": class_id

            })

    print()
    print(f"Total : {len(annotations)} polygones")
    print()

    return annotations
