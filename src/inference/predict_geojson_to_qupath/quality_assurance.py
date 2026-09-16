"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    quality_assurance.py

Description:
    Quality assessment of the generated tissue annotations.

    The module evaluates annotation characteristics and
    identifies regions that may require additional
    verification after automated prediction.

Author:
    Hamza Graïn
"""

import os
import csv
import numpy as np

from config import *


# SCORE DE QUALITE


def compute_quality_score(annotation):

    stats = annotation["statistics"]

    confidence = annotation.get("confidence", 1.0)

    score = 100

    # -------------------------------
    # Faible confiance
    # -------------------------------

    if confidence < 0.95:
        score -= 10

    if confidence < 0.90:
        score -= 15

    if confidence < 0.80:
        score -= 25

    # -------------------------------
    # Région très petite
    # -------------------------------

    if stats["area"] < QA_SMALL_REGION:
        score -= 15

    # -------------------------------
    # Région très fine
    # -------------------------------

    bbox = stats["bbox"]

    width = bbox[2] - bbox[0]

    height = bbox[3] - bbox[1]

    if min(width, height) < QA_THIN_REGION:
        score -= 20

    return max(score, 0)



# DETECTION DES PROBLEMES


def detect_warnings(annotation):

    warnings = []

    stats = annotation["statistics"]

    confidence = annotation.get("confidence", 1.0)

    bbox = stats["bbox"]

    width = bbox[2] - bbox[0]

    height = bbox[3] - bbox[1]

    # ----------------------------

    if confidence < QA_MIN_CONFIDENCE:

        warnings.append("Faible confiance")

    # ----------------------------

    if stats["area"] < QA_SMALL_REGION:

        warnings.append("Petite région")

    # ----------------------------

    if min(width, height) < QA_THIN_REGION:

        warnings.append("Région très fine")

    # ----------------------------

    if stats["perimeter"] > 10000:

        warnings.append("Contour complexe")

    return warnings



# EXPORT CSV


def export_warnings(results):

    output = os.path.join(

        OUTPUT_DIR,

        WARNINGS_NAME

    )

    with open(

        output,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "ID",

            "Classe",

            "Score",

            "Confiance",

            "Surface",

            "Périmètre",

            "Warnings"

        ])

        for r in results:

            writer.writerow([

                r["id"],

                r["class"],

                r["score"],

                f"{r['confidence']:.3f}",

                f"{r['area']:.0f}",

                f"{r['perimeter']:.1f}",

                "; ".join(r["warnings"])

            ])

    return output



# QUALITY ASSURANCE


def quality_assurance(annotations):

    print("="*60)
    print("QUALITY ASSURANCE")
    print("="*60)

    results = []

    for i, ann in enumerate(annotations):

        score = compute_quality_score(ann)

        warnings = detect_warnings(ann)

        result = {

            "id": i,

            "class": CLASS_NAMES[

                ann["class_id"]

            ],

            "score": score,

            "confidence": ann.get(

                "confidence",

                1.0

            ),

            "area": ann["statistics"]["area"],

            "perimeter": ann["statistics"]["perimeter"],

            "warnings": warnings

        }

        results.append(result)

    csv_path = export_warnings(results)

    print()

    print("Résumé QA")

    print("-"*40)

    print(

        f"Nombre d'annotations : {len(results)}"

    )

    print(

        f"Score moyen : {np.mean([r['score'] for r in results]):.1f}"

    )

    print(

        f"Annotations parfaites : {sum(r['score']==100 for r in results)}"

    )

    print(

        f"Annotations à vérifier : {sum(r['score']<80 for r in results)}"

    )

    print()

    print(csv_path)

    print()

    return results
