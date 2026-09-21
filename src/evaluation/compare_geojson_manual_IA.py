"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    compare_geojson_manual_IA.py

Description:
    Comparison of manual and AI-generated GeoJSON annotations.

    The module evaluates the spatial agreement between manual
    reference annotations and predicted tissue compartment
    annotations using geometric segmentation metrics.

Author:
    Hamza Graïn
"""

import csv
import json
import os

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid



# CONFIGURATION


MANUAL_GEOJSON = r"path\to\the\annotations_biopsie_number_manual.geojson"
AI_GEOJSON = r"path\to\the\annotations_biopsie_number_IA.geojson"
OUTPUT_CSV = r"comparaison_biopsie_number_manual_IA.csv"

# Classes à comparer. "Vide" est volontairement exclue.
CLASSES_TO_EVALUATE = ["CK14", "Stroma", "Autres"]



# LECTURE


def load_geojson(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} n'est pas un FeatureCollection GeoJSON.")

    return data["features"]


def get_class_name(feature):
    props = feature.get("properties", {})
    classification = props.get("classification", {})
    return classification.get("name")



# GEOMETRIES


def polygon_parts(geometry):
    """Conserve uniquement les Polygon/MultiPolygon."""

    if geometry.is_empty:
        return []

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if hasattr(geometry, "geoms"):
        parts = []
        for geom in geometry.geoms:
            if isinstance(geom, Polygon):
                parts.append(geom)
            elif isinstance(geom, MultiPolygon):
                parts.extend(list(geom.geoms))
        return parts

    return []


def build_class_geometries(features):
    """
    Regroupe les polygones d'une même classe par union.
    Ainsi, le nombre de polygones n'influence pas artificiellement
    les métriques de concordance.
    """

    geometries = {c: [] for c in CLASSES_TO_EVALUATE}
    counts = {c: 0 for c in CLASSES_TO_EVALUATE}

    for feature in features:
        class_name = get_class_name(feature)

        if class_name not in CLASSES_TO_EVALUATE:
            continue

        geometry = shape(feature["geometry"])
        parts = polygon_parts(geometry)

        if parts:
            counts[class_name] += 1
            geometries[class_name].extend(parts)

    unions = {}
    for class_name in CLASSES_TO_EVALUATE:
        if geometries[class_name]:
            unions[class_name] = unary_union(geometries[class_name])
        else:
            unions[class_name] = Polygon()

    return unions, counts



# METRIQUES


def compute_metrics(manual_geometry, ai_geometry):
    """
    Comparaison spatiale basée sur les surfaces :

        TP = aire(manuel ∩ IA)
        FP = aire(IA) - TP
        FN = aire(manuel) - TP

    Les coordonnées des deux GeoJSON étant dans le même système
    de coordonnées de la WSI, aucune rasterisation n'est nécessaire.
    """

    manual_area = float(manual_geometry.area)
    ai_area = float(ai_geometry.area)

    intersection_area = float(
        manual_geometry.intersection(ai_geometry).area
    )

    fp_area = max(0.0, ai_area - intersection_area)
    fn_area = max(0.0, manual_area - intersection_area)

    precision = (
        intersection_area / ai_area
        if ai_area > 0
        else (1.0 if manual_area == 0 else 0.0)
    )

    recall = (
        intersection_area / manual_area
        if manual_area > 0
        else (1.0 if ai_area == 0 else 0.0)
    )

    dice_denominator = (
        2.0 * intersection_area + fp_area + fn_area
    )
    dice = (
        2.0 * intersection_area / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    iou_denominator = (
        intersection_area + fp_area + fn_area
    )
    iou = (
        intersection_area / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    return {
        "manual_area": manual_area,
        "ai_area": ai_area,
        "intersection_area": intersection_area,
        "fp_area": fp_area,
        "fn_area": fn_area,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }



# COMPARAISON


def compare_geojson(manual_path, ai_path, output_csv):
    print()
    print("=" * 70)
    print("COMPARAISON ANNOTATIONS MANUELLES / IA")
    print("=" * 70)
    print()

    manual_features = load_geojson(manual_path)
    ai_features = load_geojson(ai_path)

    print(f"Annotations manuelles : {len(manual_features)}")
    print(f"Annotations IA        : {len(ai_features)}")
    print()

    manual_geom, manual_counts = build_class_geometries(manual_features)
    ai_geom, ai_counts = build_class_geometries(ai_features)

    results = []

    for class_name in CLASSES_TO_EVALUATE:
        m = compute_metrics(
            manual_geom[class_name],
            ai_geom[class_name]
        )

        row = {
            "Classe": class_name,
            "Annotations manuelles": manual_counts[class_name],
            "Annotations IA": ai_counts[class_name],
            "Aire manuelle": m["manual_area"],
            "Aire IA": m["ai_area"],
            "Aire intersection": m["intersection_area"],
            "FP (aire)": m["fp_area"],
            "FN (aire)": m["fn_area"],
            "Dice (%)": 100 * m["dice"],
            "IoU (%)": 100 * m["iou"],
            "Precision (%)": 100 * m["precision"],
            "Rappel (%)": 100 * m["recall"],
        }

        results.append(row)

        print("-" * 70)
        print(class_name)
        print("-" * 70)
        print(f"Annotations manuelles : {manual_counts[class_name]}")
        print(f"Annotations IA        : {ai_counts[class_name]}")
        print(f"Dice                  : {100*m['dice']:.2f} %")
        print(f"IoU                   : {100*m['iou']:.2f} %")
        print(f"Precision             : {100*m['precision']:.2f} %")
        print(f"Rappel                : {100*m['recall']:.2f} %")

    # Moyenne macro : chaque classe a le même poids
    macro = {
        "Classe": "Moyenne macro",
        "Annotations manuelles": "",
        "Annotations IA": "",
        "Aire manuelle": "",
        "Aire IA": "",
        "Aire intersection": "",
        "FP (aire)": "",
        "FN (aire)": "",
        "Dice (%)": sum(r["Dice (%)"] for r in results) / len(results),
        "IoU (%)": sum(r["IoU (%)"] for r in results) / len(results),
        "Precision (%)": sum(r["Precision (%)"] for r in results) / len(results),
        "Rappel (%)": sum(r["Rappel (%)"] for r in results) / len(results),
    }

    results.append(macro)

    print()
    print("=" * 70)
    print("MOYENNE MACRO")
    print("=" * 70)
    print(f"Dice      : {macro['Dice (%)']:.2f} %")
    print(f"IoU       : {macro['IoU (%)']:.2f} %")
    print(f"Precision : {macro['Precision (%)']:.2f} %")
    print(f"Rappel    : {macro['Rappel (%)']:.2f} %")

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"CSV sauvegardé : {output_csv}")
    print()


if __name__ == "__main__":
    compare_geojson(
        MANUAL_GEOJSON,
        AI_GEOJSON,
        OUTPUT_CSV
    )
