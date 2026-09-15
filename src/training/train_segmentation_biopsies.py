"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    train_segmentation_par_biopsie.py

Description:
    Training and validation of the semantic segmentation model
    using a biopsy-based dataset split.

    The script organizes the training and validation data by
    biopsy to evaluate the model on previously unseen biopsies.

Author:
    Hamza Graïn
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as A

from torch.cuda.amp import autocast
from torch.amp import GradScaler

from sklearn.metrics import jaccard_score


# CONFIGURATION


IMAGE_DIR = "Dataset_IA/images"
MASK_DIR = "Dataset_IA/masks"

MODEL_NAME = "best_model_unet_biopsie_number_indep.pth"

NUM_CLASSES = 4

PATCH_SIZE = 512

BATCH_SIZE = 8

EPOCHS = 300

LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 0

print("="*60)

print("DEVICE :", DEVICE)

if DEVICE == "cuda":
    print("GPU :", torch.cuda.get_device_name(0))

print("="*60)


# BIOPSIE DE VALIDATION






TEST_BIOPSY = "biopsie_number"









# DATA AUGMENTATION


train_transform = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.RandomRotate90(p=0.5),

    A.RandomBrightnessContrast(
        p=0.3
    ),

    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.3
    )

])


# DATASET


class HistologyDataset(Dataset):

    def __init__(
        self,
        image_files,
        mask_files,
        transform=None
    ):

        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        image = cv2.imread(self.image_files[idx])

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            self.mask_files[idx],
            cv2.IMREAD_GRAYSCALE
        )

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.

        image = np.transpose(image, (2,0,1))

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask
    


# LECTURE DES FICHIERS


image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
])

mask_files = sorted([
    os.path.join(MASK_DIR, f)
    for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])

print()

print("Nombre d'images :", len(image_files))
print("Nombre de masques :", len(mask_files))


# DETECTION AUTOMATIQUE DES BIOPSIES


all_biopsies = sorted({

    os.path.basename(f)
      .split("_patch")[0]
      .lower()

    for f in image_files

})

print()

print("Biopsies détectées :")

for b in all_biopsies:
    print(" -", b)



# TRAIN / VALIDATION


TRAIN_BIOPSIES = [

    b

    for b in all_biopsies

    if b != TEST_BIOPSY.lower()

]

VAL_BIOPSIES = [

    TEST_BIOPSY.lower()

]

print()

print("Biopsies Train :")

for b in TRAIN_BIOPSIES:
    print(" -", b)

print()

print("Biopsie Validation :")

for b in VAL_BIOPSIES:
    print(" -", b)



# CONSTRUCTION DES LISTES


train_imgs = []
train_masks = []

val_imgs = []
val_masks = []

for img, mask in zip(image_files, mask_files):

    filename = os.path.basename(img).lower()

    if any(b in filename for b in TRAIN_BIOPSIES):

        train_imgs.append(img)
        train_masks.append(mask)

    elif any(b in filename for b in VAL_BIOPSIES):

        val_imgs.append(img)
        val_masks.append(mask)


# AFFICHAGE


print()

print("="*60)

print("Train patches :", len(train_imgs))

print("Validation patches :", len(val_imgs))

print("="*60)


# DATASETS


train_dataset = HistologyDataset(
    train_imgs,
    train_masks,
    transform=train_transform
)

val_dataset = HistologyDataset(
    val_imgs,
    val_masks,
    transform=None
)


# DATALOADERS


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE=="cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE=="cuda")
)


# MODELE


print("\n" + "="*60)

print("Modèle : U-Net")
print("Encodeur : ResNet34")
print("Poids : ImageNet")

print("="*60)

model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights="imagenet",

    in_channels=3,

    classes=NUM_CLASSES

)

model.to(DEVICE)


# LOSS


class_weights = torch.tensor(

    [
        0.5,   # Fond
        1.0,   # CK14
        1.0,   # Stroma
        1.5    # Autres
    ],

    dtype=torch.float32,

    device=DEVICE

)

ce_loss = nn.CrossEntropyLoss(
    weight=class_weights
)

dice_loss = smp.losses.DiceLoss(
    mode="multiclass"
)

def segmentation_loss(outputs, targets):

    ce = ce_loss(outputs, targets)

    dice = dice_loss(outputs, targets)

    return 0.5 * ce + 0.5 * dice


# OPTIMIZER


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=1e-4

)


# LEARNING RATE SCHEDULER


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3,

    min_lr=1e-6

)


# MIXED PRECISION


scaler = GradScaler(
    "cuda",
    enabled=(DEVICE=="cuda")
)


# EARLY STOPPING


best_val_loss = np.inf

patience = 8

counter = 0


# HISTORIQUE


history_train = []

history_val = []

history_lr = []


# NOM DES CLASSES


CLASS_NAMES = [

    "Fond",

    "CK14",

    "Stroma",

    "Autres"

]


# IoU


def compute_iou(pred, target, cls):

    pred = pred == cls

    target = target == cls

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    union = np.logical_or(
        pred,
        target
    ).sum()

    if union == 0:
        return np.nan

    return intersection / union


# DICE


def compute_dice(pred, target, cls):

    pred = pred == cls

    target = target == cls

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    total = pred.sum() + target.sum()

    if total == 0:
        return np.nan

    return 2 * intersection / total


# ENTRAINEMENT


print("\nDébut de l'entraînement...\n")

for epoch in range(EPOCHS):

    print("=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    
    # TRAIN
    

    model.train()

    train_loss = 0.0

    train_iou = [[] for _ in range(NUM_CLASSES)]
    train_dice = [[] for _ in range(NUM_CLASSES)]

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, masks in loop:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
             "cuda",
             enabled=(DEVICE=="cuda")
        ):

            outputs = model(images)

            loss = segmentation_loss(
                outputs,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        
        # METRIQUES TRAIN
     

        preds = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        gts = masks.cpu().numpy()

        for pred, gt in zip(preds, gts):

            for cls in range(NUM_CLASSES):

                iou = compute_iou(
                    pred,
                    gt,
                    cls
                )

                dice = compute_dice(
                    pred,
                    gt,
                    cls
                )

                if not np.isnan(iou):
                    train_iou[cls].append(iou)

                if not np.isnan(dice):
                    train_dice[cls].append(dice)

        loop.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_loader)

    
    # VALIDATION
    

    model.eval()

    val_loss = 0.0

    val_iou = [[] for _ in range(NUM_CLASSES)]
    val_dice = [[] for _ in range(NUM_CLASSES)]

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(enabled=(DEVICE=="cuda")):

                outputs = model(images)

                loss = segmentation_loss(
                    outputs,
                    masks
                )

            val_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

            gts = masks.cpu().numpy()

            for pred, gt in zip(preds, gts):

                for cls in range(NUM_CLASSES):

                    iou = compute_iou(
                        pred,
                        gt,
                        cls
                    )

                    dice = compute_dice(
                        pred,
                        gt,
                        cls
                    )

                    if not np.isnan(iou):
                        val_iou[cls].append(iou)

                    if not np.isnan(dice):
                        val_dice[cls].append(dice)

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    history_train.append(train_loss)
    history_val.append(val_loss)

    history_lr.append(
        optimizer.param_groups[0]["lr"]
    )

    
    # AFFICHAGE
    

    print()

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print()

    print(
        f"{'Classe':<12}"
        f"{'IoU':>10}"
        f"{'Dice':>10}"
    )

    print("-"*34)

    mean_iou = []

    mean_dice = []

    for cls in range(NUM_CLASSES):

        miou = np.mean(val_iou[cls])

        mdice = np.mean(val_dice[cls])

        mean_iou.append(miou)

        mean_dice.append(mdice)

        print(
            f"{CLASS_NAMES[cls]:<12}"
            f"{miou:>10.3f}"
            f"{mdice:>10.3f}"
        )

    print("-"*34)

    print(
        f"{'mIoU':<12}"
        f"{np.nanmean(mean_iou):>10.3f}"
    )

    print(
        f"{'mDice':<12}"
        f"{np.nanmean(mean_dice):>10.3f}"
    )

    
    # SAUVEGARDE
    

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            MODEL_NAME
        )

        print("\n>>> Nouveau meilleur modèle sauvegardé")

    else:

        counter += 1

        print(
            f"\nEarly Stopping : {counter}/{patience}"
        )

    ###########################################################
    # EARLY STOPPING
    ###########################################################

    if counter >= patience:

        print("\nArrêt anticipé.")

        break

    
# FIN DE L'ENTRAINEMENT


print("\n")
print("=" * 70)
print("ENTRAINEMENT TERMINE")
print("=" * 70)

print(f"Meilleure Validation Loss : {best_val_loss:.4f}")


# COURBE DES LOSSES


plt.figure(figsize=(10,5))

plt.plot(
    history_train,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    history_val,
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Learning Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "learning_curve.png",
    dpi=300
)

plt.show()


# COURBE DU LEARNING RATE


plt.figure(figsize=(10,5))

plt.plot(
    history_lr,
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Learning Rate")

plt.title("Learning Rate Schedule")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "learning_rate.png",
    dpi=300
)

plt.show()


# RECHARGEMENT DU MEILLEUR MODELE


print()

print("Chargement du meilleur modèle...")

model.load_state_dict(

    torch.load(

        MODEL_NAME,

        map_location=DEVICE

    )

)

model.eval()

print("Meilleur modèle chargé.")


# RESUME


print()

print("="*70)

print("Résumé")

print("="*70)

print()

print("Train biopsies :")

for b in TRAIN_BIOPSIES:
    print(" -", b)

print()

print("Validation :")

for b in VAL_BIOPSIES:
    print(" -", b)

print()

print("Nombre de patches train :", len(train_dataset))

print("Nombre de patches validation :", len(val_dataset))

print()

print("Batch Size :", BATCH_SIZE)

print("Epochs max :", EPOCHS)

print("Learning Rate :", LR)

print()

print("GPU :", DEVICE)

if DEVICE == "cuda":

    print(
        torch.cuda.get_device_name(0)
    )

print()

print("Meilleure Validation Loss :", best_val_loss)

print()

print("Modèle sauvegardé :", MODEL_NAME)

print()

print("Figures sauvegardées :")

print(" - learning_curve.png")

print(" - learning_rate.png")

print()

print("="*70)


# EVALUATION FINALE


print("\n")
print("=" * 70)
print("EVALUATION FINALE")
print("=" * 70)

model.eval()

final_iou = [[] for _ in range(NUM_CLASSES)]
final_dice = [[] for _ in range(NUM_CLASSES)]

with torch.no_grad():

    for images, masks in tqdm(
        val_loader,
        desc="Evaluation"
    ):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        gts = masks.cpu().numpy()

        for pred, gt in zip(preds, gts):

            for cls in range(NUM_CLASSES):

                iou = compute_iou(
                    pred,
                    gt,
                    cls
                )

                dice = compute_dice(
                    pred,
                    gt,
                    cls
                )

                if not np.isnan(iou):
                    final_iou[cls].append(iou)

                if not np.isnan(dice):
                    final_dice[cls].append(dice)

print()

print("="*45)

print(
    f"{'Classe':<12}"
    f"{'IoU':>10}"
    f"{'Dice':>10}"
)

print("="*45)

for cls in range(NUM_CLASSES):

    print(

        f"{CLASS_NAMES[cls]:<12}"

        f"{np.mean(final_iou[cls]):>10.3f}"

        f"{np.mean(final_dice[cls]):>10.3f}"

    )

print("="*45)

print()

print("Evaluation terminée.")
