"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    distribution_compartiments.py

Description:
    Analysis of tissue compartment distributions within
    annotated biopsies.

    The module computes the proportion of each tissue class
    and summarizes the compartment composition of the samples.

Author:
    Hamza Graïn
"""

import cv2
import numpy as np
import os

counts = {}

for file in os.listdir("Dataset_IA/masks"):

    mask = cv2.imread(
        os.path.join("Dataset_IA/masks", file),
        cv2.IMREAD_GRAYSCALE
    )

    unique, c = np.unique(mask, return_counts=True)

    for u, n in zip(unique, c):
        counts[u] = counts.get(u, 0) + n

total = sum(counts.values())

for cls in sorted(counts):
    print(
        f"Classe {cls}: "
        f"{100*counts[cls]/total:.2f}%"
    )
