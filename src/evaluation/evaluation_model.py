"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    evaluation_model.py

Description:
    Evaluation of the trained semantic segmentation model.

    The module computes segmentation performance metrics
    for the different tissue classes and summarizes the
    overall model performance.

Author:
    Hamza Graïn
"""

import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



### CONFIGURATION


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best_model_unet.pth"

IMAGE_DIR = "Dataset_IA/images"
MASK_DIR = "Dataset_IA/masks"

NUM_CLASSES = 4

CLASS_NAMES = [
    "Vide",
    "CK14",
    "Stroma",
    "Autres"
]

# Nombre de patches à évaluer
NUM_SAMPLES = 300

# Graine aléatoire pour utiliser toujours
# le même échantillon de patches
RANDOM_SEED = 42


### CHARGEMENT DU MODELE


print(f"Utilisation du périphérique : {DEVICE}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Modèle chargé avec succès.")



### METRIQUES


def compute_iou(pred, target, cls):

    pred_cls = pred == cls
    target_cls = target == cls

    intersection = np.logical_and(
        pred_cls,
        target_cls
    ).sum()

    union = np.logical_or(
        pred_cls,
        target_cls
    ).sum()

    if union == 0:
        return np.nan

    return intersection / union


def compute_dice(pred, target, cls):

    pred_cls = pred == cls
    target_cls = target == cls

    intersection = np.logical_and(
        pred_cls,
        target_cls
    ).sum()

    denominator = (
        pred_cls.sum()
        +
        target_cls.sum()
    )

    if denominator == 0:
        return np.nan

    return (
        2 * intersection
        /
        denominator
    )


### SELECTION DES PATCHES


all_files = sorted(
    os.listdir(IMAGE_DIR)
)

# Vérification qu'il y a bien des fichiers
if len(all_files) == 0:

    raise RuntimeError(
        "Aucun fichier trouvé dans IMAGE_DIR."
    )


# Graine aléatoire
np.random.seed(RANDOM_SEED)


# Sélection aléatoire de NUM_SAMPLES patches
if len(all_files) > NUM_SAMPLES:

    files = list(
        np.random.choice(
            all_files,
            size=NUM_SAMPLES,
            replace=False
        )
    )

else:

    files = all_files


print("=" * 60)
print(
    f"Nombre total de patches disponibles : "
    f"{len(all_files)}"
)
print(
    f"Nombre de patches sélectionnés pour l'évaluation : "
    f"{len(files)}"
)
print("=" * 60)



### EVALUATION


ious_per_class = [
    [] for _ in range(NUM_CLASSES)
]

dices_per_class = [
    [] for _ in range(NUM_CLASSES)
]


# Matrice de confusion cumulative
cm = np.zeros(
    (
        NUM_CLASSES,
        NUM_CLASSES
    ),
    dtype=np.int64
)


for i, filename in enumerate(files):

    print(
        f"Évaluation du patch "
        f"{i + 1}/{len(files)} : "
        f"{filename}"
    )


    
    ### CHARGEMENT DE L'IMAGE
    

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            f"Impossible de lire l'image : "
            f"{filename}"
        )

        continue


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


   
    # CHARGEMENT DU MASQUE
    

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        print(
            f"Impossible de lire le masque : "
            f"{filename}"
        )

        continue


    
    ### PREPARATION DE L'IMAGE
    

    x = image.astype(
        np.float32
    ) / 255.0


    # H, W, C -> C, H, W
    x = np.transpose(
        x,
        (2, 0, 1)
    )


    x = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)


    
    ### PREDICTION
    

    with torch.no_grad():

        output = model(x)


    pred = torch.argmax(
        output,
        dim=1
    ).cpu().numpy()[0]


    
    ### MISE A JOUR DE LA MATRICE DE CONFUSION
    

    cm += confusion_matrix(
        mask.flatten(),
        pred.flatten(),
        labels=list(
            range(NUM_CLASSES)
        )
    )


    
    ### CALCUL DES METRIQUES PAR CLASSE
    

    for cls in range(NUM_CLASSES):

        iou = compute_iou(
            pred,
            mask,
            cls
        )

        dice = compute_dice(
            pred,
            mask,
            cls
        )


        if not np.isnan(iou):

            ious_per_class[cls].append(
                iou
            )


        if not np.isnan(dice):

            dices_per_class[cls].append(
                dice
            )



### RESULTATS


print("\n")
print("=" * 60)
print("RESULTATS DE L'EVALUATION")
print("=" * 60)


mean_ious = []
mean_dices = []


for cls in range(NUM_CLASSES):

    if len(ious_per_class[cls]) > 0:

        miou = np.mean(
            ious_per_class[cls]
        )

    else:

        miou = np.nan


    if len(dices_per_class[cls]) > 0:

        mdice = np.mean(
            dices_per_class[cls]
        )

    else:

        mdice = np.nan


    mean_ious.append(miou)
    mean_dices.append(mdice)


    print(
        f"{CLASS_NAMES[cls]:10s}"
        f" | IoU = {miou:.4f}"
        f" | Dice = {mdice:.4f}"
    )


print("=" * 60)

print(
    f"mIoU global : "
    f"{np.nanmean(mean_ious):.4f}"
)

print(
    f"Dice global : "
    f"{np.nanmean(mean_dices):.4f}"
)

print("=" * 60)



### MATRICE DE CONFUSION


print("\n")
print("=" * 60)
print("MATRICE DE CONFUSION")
print("=" * 60)

print(cm)



### VISUALISATION D'UN PATCH


# Choix d'un patch parmi ceux sélectionnés
idx = 1

# Vérification que l'index existe
if idx >= len(files):

    idx = 0


filename = files[idx]


print("\n")
print(
    f"Visualisation du patch : "
    f"{filename}"
)


# Chargement image
image = cv2.imread(
    os.path.join(
        IMAGE_DIR,
        filename
    )
)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# Chargement masque
mask = cv2.imread(
    os.path.join(
        MASK_DIR,
        filename
    ),
    cv2.IMREAD_GRAYSCALE
)


# Préparation image
x = image.astype(
    np.float32
) / 255.0


x = np.transpose(
    x,
    (2, 0, 1)
)


x = torch.tensor(
    x,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)


# Prédiction
with torch.no_grad():

    output = model(x)


pred = torch.argmax(
    output,
    dim=1
).cpu().numpy()[0]



### AFFICHAGE

plt.figure(
    figsize=(16, 5)
)


# Image originale
plt.subplot(1, 3, 1)

plt.imshow(image)

plt.title(
    "Image originale"
)

plt.axis("off")


# Masque réel
plt.subplot(1, 3, 2)

plt.imshow(
    mask,
    cmap="jet",
    vmin=0,
    vmax=NUM_CLASSES - 1
)

plt.title(
    "Masque réel"
)

plt.axis("off")


# Prédiction
plt.subplot(1, 3, 3)

plt.imshow(
    pred,
    cmap="jet",
    vmin=0,
    vmax=NUM_CLASSES - 1
)

plt.title(
    "Prédiction du modèle"
)

plt.axis("off")


plt.tight_layout()

plt.show()
