"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    polygon_processing.py

Description:
    Geometric validation and processing of segmentation
    polygons before QuPath export.

    The module validates, simplifies, filters, and merges
    polygon geometries and computes their associated statistics.

Author:
    Hamza Graïn
"""

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from config import *


# VALIDATION


def validate_polygon(poly):

    if poly.is_empty:
        return None

    # Corrige les auto-intersections
    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        return None

    # Supprime les petits objets
    if poly.area < MIN_REGION_AREA:
        return None

    return poly



# SIMPLIFICATION


def simplify_polygon(poly):

    poly = poly.simplify(

        SIMPLIFICATION,

        preserve_topology=True

    )

    return poly



# FUSION DES POLYGONES PROCHES


def merge_polygons(polygons):

    if len(polygons) == 0:
        return []

    buffered = [

        p.buffer(MERGE_DISTANCE)

        for p in polygons

    ]

    merged = unary_union(buffered)

    merged = merged.buffer(-MERGE_DISTANCE)

    if merged.is_empty:
        return []

    if isinstance(merged, Polygon):
        return [merged]

    if isinstance(merged, MultiPolygon):
        return list(merged.geoms)

    return []



# STATISTIQUES


def compute_statistics(poly):

    xmin, ymin, xmax, ymax = poly.bounds

    return {

        "area": float(poly.area),

        "perimeter": float(poly.length),

        "bbox": [

            xmin,

            ymin,

            xmax,

            ymax

        ]

    }



# TRAITEMENT D'UNE CLASSE


def process_class(polygons):

    clean = []

    for poly in polygons:

        poly = validate_polygon(poly)

        if poly is None:
            continue

        poly = simplify_polygon(poly)

        clean.append(poly)

    clean = merge_polygons(clean)

    return clean



# PIPELINE GLOBAL


def process_all(annotations):

    print("=" * 60)
    print("Traitement géométrique")
    print("=" * 60)

    output = []

    for class_id in range(1, NUM_CLASSES):

        class_polygons = [

            ann["polygon"]

            for ann in annotations

            if ann["class_id"] == class_id

        ]

        class_polygons = process_class(

            class_polygons

        )

        print(

            f"{CLASS_NAMES[class_id]} : {len(class_polygons)} polygones"

        )

        for poly in class_polygons:

            output.append({

                "polygon": poly,

                "class_id": class_id,

                "statistics": compute_statistics(poly),

                "confidence": 1.0

            })

    print()

    print(f"Total final : {len(output)} annotations")

    print()

    return output
