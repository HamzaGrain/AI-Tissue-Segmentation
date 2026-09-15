"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    train_segmentation_par_patchs.py

Description:
    Training of the semantic segmentation model using
    image patches and their corresponding ground-truth masks.

    The script trains and validates the model on the prepared
    patch-based dataset and saves the best-performing model
    according to the validation performance.

Author:
    Hamza Graïn
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A


# CONFIGURATION


IMAGE_DIR = "Dataset_IA/images"
MASK_DIR = "Dataset_IA/masks"

MODEL_SAVE = "best_model_unet.pth"

NUM_CLASSES = 4

CLASS_NAMES = [
    "Fond",
    "CK14",
    "Stroma",
    "Autres"
]

PATCH_SIZE = 512

BATCH_SIZE = 8

EPOCHS = 300

LR = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("DEVICE :", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )

print("=" * 60)


# REPRODUCTIBILITE


SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# DATA AUGMENTATION


train_transform = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.RandomRotate90(p=0.5),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    ),

    A.HueSaturationValue(
        hue_shift_limit=5,
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.3
    ),

])

val_transform = None


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

        image = cv2.imread(
            self.image_files[idx]
        )

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

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask


# CHARGEMENT DES DONNEES


image_files = sorted([

    os.path.join(
        IMAGE_DIR,
        f
    )

    for f in os.listdir(IMAGE_DIR)

    if f.endswith(".png")

])

mask_files = sorted([

    os.path.join(
        MASK_DIR,
        f
    )

    for f in os.listdir(MASK_DIR)

    if f.endswith(".png")

])

print()

print("Nombre d'images :", len(image_files))

print("Nombre de masques :", len(mask_files))

train_imgs, val_imgs, train_masks, val_masks = train_test_split(

    image_files,
    mask_files,

    test_size=0.20,

    random_state=SEED,

    shuffle=True

)

train_dataset = HistologyDataset(

    train_imgs,
    train_masks,
    transform=train_transform

)

val_dataset = HistologyDataset(

    val_imgs,
    val_masks,
    transform=val_transform

)



# DATALOADER


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Train patches : {len(train_dataset)}")
print(f"Validation patches : {len(val_dataset)}")


# MODELE


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
)

model = model.to(DEVICE)

print()
print("="*60)
print("Modèle : U-Net")
print("Encodeur : ResNet34")
print("Poids : ImageNet")
print("="*60)


# LOSS


# Le fond est légèrement moins important
class_weights = torch.tensor(
    [
        0.5,   # Fond
        1.0,   # CK14
        1.0,   # Stroma
        1.2    # Autres
    ],
    dtype=torch.float32,
    device=DEVICE
)

ce_loss = nn.CrossEntropyLoss(
    weight=class_weights
)

dice_loss = smp.losses.DiceLoss(
    mode="multiclass",
    from_logits=True
)


# LOSS FINALE


def segmentation_loss(outputs, masks):

    ce = ce_loss(outputs, masks)

    dice = dice_loss(outputs, masks)

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


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# EARLY STOPPING


best_val_loss = np.inf

patience = 10

counter = 0


# HISTORIQUE


history_train = []

history_val = []

history_lr = []


# METRIQUES


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


def compute_dice(pred, target, cls):

    pred = pred == cls

    target = target == cls

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    denom = pred.sum() + target.sum()

    if denom == 0:

        return np.nan

    return 2 * intersection / denom


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

    loop = tqdm(train_loader)

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
            device_type="cuda",
            enabled=torch.cuda.is_available()
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
        ).detach().cpu().numpy()

        gts = masks.detach().cpu().numpy()

        for p, gt in zip(preds, gts):

            for cls in range(NUM_CLASSES):

                iou = compute_iou(
                    p,
                    gt,
                    cls
                )

                dice = compute_dice(
                    p,
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

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

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

            for p, gt in zip(preds, gts):

                for cls in range(NUM_CLASSES):

                    iou = compute_iou(
                        p,
                        gt,
                        cls
                    )

                    dice = compute_dice(
                        p,
                        gt,
                        cls
                    )

                    if not np.isnan(iou):
                        val_iou[cls].append(iou)

                    if not np.isnan(dice):
                        val_dice[cls].append(dice)

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history_train.append(train_loss)
    history_val.append(val_loss)
    history_lr.append(current_lr)

   
    # AFFICHAGE DES METRIQUES
    

    print()

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Learning Rate : {current_lr:.2e}"
    )

    print()

    print("-" * 60)

    print(
        f"{'Classe':<12}"
        f"{'IoU':>10}"
        f"{'Dice':>10}"
    )

    print("-" * 60)

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

    print("-" * 60)

    print(
        f"mIoU : {np.nanmean(mean_iou):.3f}"
    )

    print(
        f"Dice : {np.nanmean(mean_dice):.3f}"
    )

    
    # SAUVEGARDE
    

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            MODEL_SAVE
        )

        print("\n>>> Nouveau meilleur modèle sauvegardé")

    else:

        counter += 1

        print(
            f"\nEarly stopping : {counter}/{patience}"
        )

    
    # EARLY STOPPING
    

    if counter >= patience:

        print("\nArrêt anticipé.")

        break



# FIN DE L'ENTRAINEMENT


print("\n")
print("=" * 70)
print("ENTRAINEMENT TERMINE")
print("=" * 70)

print(f"Meilleure Validation Loss : {best_val_loss:.4f}")


# COURBES


plt.figure(figsize=(10, 5))

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



plt.figure(figsize=(10,5))

plt.plot(
    history_lr,
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Learning Rate")

plt.title("Learning Rate")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "learning_rate.png",
    dpi=300
)

plt.show()


# RECHARGEMENT DU MEILLEUR MODELE


print("\nChargement du meilleur modèle...")

model.load_state_dict(
    torch.load(
        MODEL_SAVE,
        map_location=DEVICE
    )
)

model.eval()

print("Modèle chargé avec succès.")


# RESUME


print("\n")
print("=" * 70)

print("Résumé de l'entraînement")

print("=" * 70)

print(f"Images          : {len(image_files)}")
print(f"Train           : {len(train_dataset)}")
print(f"Validation      : {len(val_dataset)}")

print(f"\nBatch Size      : {BATCH_SIZE}")
print(f"Epoch max       : {EPOCHS}")
print(f"Learning Rate   : {LR}")

print(f"\nGPU             : {DEVICE}")

if torch.cuda.is_available():

    print(
        "Nom GPU         :",
        torch.cuda.get_device_name(0)
    )

print(f"\nMeilleure Loss  : {best_val_loss:.4f}")

print(f"\nModèle sauvegardé : {MODEL_SAVE}")

print("\nFigures sauvegardées :")

print(" - learning_curve.png")

print(" - learning_rate.png")

print("=" * 70)

print("\nProjet terminé avec succès.")
