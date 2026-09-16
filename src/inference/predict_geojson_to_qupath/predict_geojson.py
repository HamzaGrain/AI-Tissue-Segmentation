"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    predict_geojson.py

Description:
    Main pipeline for automated tissue compartment
    segmentation and QuPath annotation generation.

    The module coordinates model inference, prediction
    post-processing, polygon extraction and processing,
    GeoJSON export, visualization, statistics, and
    quality assurance.

Author:
    Hamza Graïn
"""

import time

from model import (
    load_model,
    load_slide,
    print_summary
)

from inference import predict_wsi

from postprocessing import clean_prediction

from polygon_extraction import extract_all_polygons

from polygon_processing import process_all

from geojson_export import export_geojson

from overlay import export_visualizations

from statistics import compute_statistics

from quality_assurance import quality_assurance


def main():

    start = time.time()

    print()
    print("=" * 70)
    print("SEGMENTATION AUTOMATIQUE DES COMPARTIMENTS")
    print("=" * 70)
    print()

    
    # Chargement
    

    model = load_model()

    slide_info = load_slide()

    print_summary(slide_info)

    
    # Inférence
    

    results = predict_wsi(

        model,

        slide_info

    )


    import cv2
    #import numpy as np


    #print(np.unique(results["prediction"]))

    #cv2.imwrite(
    #        "RAW_PREDICTION.png",
     #       results["prediction"]
    #)

    # ======================================================
    # Post-processing
    # ======================================================

    clean = clean_prediction(results)

    
    # Extraction des polygones
    

    polygons = extract_all_polygons(clean)

    
    # Traitement géométrique
    

    annotations = process_all(polygons)

    
    # Export GeoJSON
    

    export_geojson(

        annotations,

        slide_info

    )

    
    # Visualisations
    
    cv2.imwrite("raw_prediction.png", results["prediction"])
    cv2.imwrite("clean_prediction.png", clean_prediction(results))
    
    export_visualizations(

        slide_info,

        results["prediction"],

        clean,

        results["confidence"]

    )

    
    # Statistiques
    

    compute_statistics(

        annotations

    )

    
    # Quality Assurance
    

    quality_assurance(

        annotations

    )

   

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("PIPELINE TERMINÉ")
    print("=" * 70)

    print(f"Temps total : {elapsed:.1f} secondes")

    print("=" * 70)


# ==========================================================

if __name__ == "__main__":

    main()
