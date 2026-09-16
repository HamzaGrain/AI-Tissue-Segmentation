"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    statistics.py

Description:
    Computation of statistics associated with the generated
    tissue compartment annotations.

    The module summarizes geometric and class-related
    characteristics of the predicted annotations.

Author:
    Hamza Graïn
"""

import os
import csv
import numpy as np

from config import *


# STATISTIQUES D'UNE CLASSE


def compute_class_statistics(annotations, class_id):

    class_annotations = [

        a

        for a in annotations

        if a["class_id"] == class_id

    ]

    if len(class_annotations) == 0:

        return {

            "count":0,

            "total_area":0,

            "mean_area":0,

            "median_area":0,

            "max_area":0,

            "min_area":0,

            "mean_perimeter":0,

            "mean_confidence":0

        }

    areas = np.array([

        a["statistics"]["area"]

        for a in class_annotations

    ])

    perimeters = np.array([

        a["statistics"]["perimeter"]

        for a in class_annotations

    ])

    confidences = np.array([

        a.get("confidence",1)

        for a in class_annotations

    ])

    return {

        "count":len(class_annotations),

        "total_area":float(np.sum(areas)),

        "mean_area":float(np.mean(areas)),

        "median_area":float(np.median(areas)),

        "max_area":float(np.max(areas)),

        "min_area":float(np.min(areas)),

        "mean_perimeter":float(np.mean(perimeters)),

        "mean_confidence":float(np.mean(confidences))

    }



# DISTRIBUTION DES COMPARTIMENTS


def compute_distribution(annotations):

    total = 0

    surfaces = {}

    for cls in CLASS_NAMES:

        if cls == 0:
            continue

        area = sum(

            a["statistics"]["area"]

            for a in annotations

            if a["class_id"] == cls

        )

        surfaces[cls] = area

        total += area

    distribution = {}

    for cls in surfaces:

        if total == 0:

            distribution[cls] = 0

        else:

            distribution[cls] = (

                100 *

                surfaces[cls]

                /

                total

            )

    return distribution



# EXPORT CSV


def export_csv(statistics, distribution):

    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True

    )

    output = os.path.join(

        OUTPUT_DIR,

        STATISTICS_NAME

    )

    with open(

        output,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "Classe",

            "Nb régions",

            "Surface totale",

            "Surface moyenne",

            "Surface médiane",

            "Surface max",

            "Surface min",

            "Périmètre moyen",

            "Confiance moyenne",

            "Distribution (%)"

        ])

        for cls in CLASS_NAMES:

            if cls == 0:
                continue

            s = statistics[cls]

            writer.writerow([

                CLASS_NAMES[cls],

                s["count"],

                s["total_area"],

                s["mean_area"],

                s["median_area"],

                s["max_area"],

                s["min_area"],

                s["mean_perimeter"],

                s["mean_confidence"],

                distribution[cls]

            ])

    return output



# RAPPORT TEXTE


def export_report(statistics, distribution):

    output = os.path.join(

        OUTPUT_DIR,

        REPORT_NAME

    )

    with open(

        output,

        "w",

        encoding="utf-8"

    ) as f:

        f.write("="*60+"\n")

        f.write("STATISTIQUES DES ANNOTATIONS\n")

        f.write("="*60+"\n\n")

        for cls in CLASS_NAMES:

            if cls == 0:
                continue

            s = statistics[cls]

            f.write(f"{CLASS_NAMES[cls]}\n")

            f.write("-"*40+"\n")

            f.write(f"Nombre régions : {s['count']}\n")

            f.write(f"Surface totale : {s['total_area']:.0f}\n")

            f.write(f"Surface moyenne : {s['mean_area']:.1f}\n")

            f.write(f"Surface médiane : {s['median_area']:.1f}\n")

            f.write(f"Surface max : {s['max_area']:.1f}\n")

            f.write(f"Surface min : {s['min_area']:.1f}\n")

            f.write(f"Périmètre moyen : {s['mean_perimeter']:.1f}\n")

            f.write(f"Confiance moyenne : {s['mean_confidence']:.3f}\n")

            f.write(f"Distribution : {distribution[cls]:.2f}%\n\n")

    return output



# CALCUL GLOBAL


def compute_statistics(annotations):

    print("="*60)

    print("Calcul des statistiques")

    print("="*60)

    statistics = {}

    for cls in CLASS_NAMES:

        if cls == 0:
            continue

        statistics[cls] = compute_class_statistics(

            annotations,

            cls

        )

    distribution = compute_distribution(

        annotations

    )

    csv_path = export_csv(

        statistics,

        distribution

    )

    report_path = export_report(

        statistics,

        distribution

    )

    print()

    print("Distribution des compartiments")

    print("-"*50)

    for cls in distribution:

        print(

            f"{CLASS_NAMES[cls]:10s} : {distribution[cls]:6.2f}%"

        )

    print()

    print(csv_path)

    print(report_path)

    print()

    return statistics, distribution
