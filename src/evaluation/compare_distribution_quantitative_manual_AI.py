"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    compare_distribution_quantitative.py

Description:
    Comparison of tissue compartment distributions between
    manual and AI-generated annotations.

    The module evaluates the quantitative agreement between
    manual reference proportions and predicted tissue
    compartment proportions.

Author:
    Hamza Graïn
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


### CONFIGURATION


GROUND_TRUTH = "path\to\your\mask_manual.png"

PREDICTION = "path\to\your\mask_AI.png"

CLASS_NAMES = {
    1: "CK14",
    2: "Stroma",
    3: "Autres"
}



### CHARGEMENT


gt = cv2.imread(GROUND_TRUTH, cv2.IMREAD_GRAYSCALE)
pred = cv2.imread(PREDICTION, cv2.IMREAD_GRAYSCALE)

print("Ground Truth :", gt.shape)
print("Prediction   :", pred.shape)

if gt is None:
    raise FileNotFoundError(GROUND_TRUTH)

if pred is None:
    raise FileNotFoundError(PREDICTION)

if gt.shape != pred.shape:
    raise ValueError("Les deux masques n'ont pas la même taille.")


### MASQUE TISSULAIRE


# Tous les pixels différents du fond
tissue = gt != 0

total_pixels = np.sum(tissue)

print()
print("="*60)
print("Distribution des compartiments")
print("="*60)
print()

print(f"Nombre de pixels du tissu : {total_pixels:,}")
print()


### CALCUL DES POURCENTAGES


gt_percentages = []
pred_percentages = []

print(f"{'Classe':<12}{'Manuel':>12}{'IA':>12}{'Différence':>15}")

print("-"*55)

for cls in CLASS_NAMES:

    gt_pixels = np.sum((gt == cls) & tissue)
    pred_pixels = np.sum((pred == cls) & tissue)

    gt_pct = 100 * gt_pixels / total_pixels
    pred_pct = 100 * pred_pixels / total_pixels

    diff = pred_pct - gt_pct

    gt_percentages.append(gt_pct)
    pred_percentages.append(pred_pct)

    print(
        f"{CLASS_NAMES[cls]:<12}"
        f"{gt_pct:>10.2f}%"
        f"{pred_pct:>10.2f}%"
        f"{diff:>13.2f}%"
    )

print("-"*55)

mean_error = np.mean(
    np.abs(
        np.array(gt_percentages) -
        np.array(pred_percentages)
    )
)

print()

print(f"Erreur moyenne : {mean_error:.2f}%")


### GRAPHIQUE


x = np.arange(len(CLASS_NAMES))

width = 0.35

plt.figure(figsize=(8,5))

plt.bar(
    x-width/2,
    gt_percentages,
    width,
    label="Manuel"
)

plt.bar(
    x+width/2,
    pred_percentages,
    width,
    label="IA"
)

plt.xticks(
    x,
    list(CLASS_NAMES.values())
)

plt.ylabel("Pourcentage (%)")

plt.title("Distribution des compartiments")

plt.legend()

plt.tight_layout()

plt.savefig(
    "compare_distribution.png",
    dpi=300
)

plt.show()

print()

print("Graphique sauvegardé : compare_distribution.png")
