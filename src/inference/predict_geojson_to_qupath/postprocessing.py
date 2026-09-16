"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    postprocessing.py

Description:
    Post-processing of the segmentation predictions.

    The module applies confidence filtering, morphological
    operations, removal of small regions, and other cleaning
    operations to improve the predicted segmentation masks.

Author:
    Hamza Graïn
"""

import cv2
import numpy as np

from config import *

# FILTRE DE CONFIANCE


def confidence_filter(
    prediction,
    confidence
):

    pred = prediction.copy()

    pred[
        confidence < MIN_CONFIDENCE
    ] = 0

    return pred



# OUVERTURE + FERMETURE


def morphology(binary):

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (
            SMOOTH_KERNEL,
            SMOOTH_KERNEL
        )

    )

    binary = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        kernel

    )

    binary = cv2.morphologyEx(

        binary,

        cv2.MORPH_CLOSE,

        kernel

    )

    return binary



# REMPLISSAGE DES TROUS


def fill_holes(binary):

    contours, hierarchy = cv2.findContours(

        binary,

        cv2.RETR_CCOMP,

        cv2.CHAIN_APPROX_SIMPLE

    )

    result = np.zeros_like(binary)

    if hierarchy is None:
        return binary

    hierarchy = hierarchy[0]

    for i, contour in enumerate(contours):

        cv2.drawContours(

            result,

            contours,

            i,

            255,

            -1

        )

    return result



# SUPPRESSION DES PETITES REGIONS


def remove_small_regions(binary):

    nb_labels, labels, stats, _ = cv2.connectedComponentsWithStats(

        binary,

        connectivity=8

    )

    result = np.zeros_like(binary)

    for i in range(1, nb_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_REGION_AREA:
            continue

        result[
            labels == i
        ] = 255

    return result



# SUPPRESSION DES REGIONS TROP FINES


def remove_thin_regions(binary):

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    result = np.zeros_like(binary)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_REGION_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if min(w, h) < QA_THIN_REGION:
            continue

        cv2.drawContours(

            result,

            [contour],

            -1,

            255,

            -1

        )

    return result



# LISSAGE


def smooth(binary):

    binary = cv2.GaussianBlur(

        binary,

        (5,5),

        0

    )

    _, binary = cv2.threshold(

        binary,

        127,

        255,

        cv2.THRESH_BINARY

    )

    return binary


# TRAITEMENT D'UNE CLASSE


def process_class(binary):

    binary = morphology(binary)

    #binary = fill_holes(binary)

    binary = remove_small_regions(binary)

    #binary = remove_thin_regions(binary)

    #binary = smooth(binary)

    return binary



# POST PROCESSING COMPLET


def clean_prediction(results):

    prediction = results["prediction"]

    confidence = results["confidence"]

    prediction = confidence_filter(

        prediction,

        confidence

    )

    clean = np.zeros_like(prediction)

    print("="*60)
    print("Post-processing")
    print("="*60)

    for cls in CLASS_NAMES:

        if cls == 0:
            continue

        print(
            f"Traitement : {CLASS_NAMES[cls]}"
        )

        binary = (

            prediction == cls

        ).astype(np.uint8)

        binary *= 255

        binary = process_class(binary)

        clean[
            binary > 0
        ] = cls

    print()

    print("Post-processing terminé.")

    print()

    return clean
