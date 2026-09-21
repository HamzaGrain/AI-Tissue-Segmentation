# Training

## Overview

This directory contains the scripts used to train the semantic
segmentation models for the automated annotation of histological
tissue compartments.

The training pipeline uses image patches and their corresponding
ground-truth segmentation masks generated during the data
preparation stage.

Two training approaches are provided:

- patch-based training
- biopsy-based training

The trained models are subsequently used during the inference
stage to segment new Whole Slide Images.

---

## Training workflow

The complete training workflow is:

```text
Prepared image / mask dataset
              │
              ▼
        Dataset loading
              │
              ▼
       Data preprocessing
              │
              ▼
        Data augmentation
              │
              ▼
        Model initialization
              │
              ▼
           Training
              │
              ▼
         Validation
              │
              ▼
      Model evaluation
              │
              ▼
       Trained model
              │
              ▼
          Inference
```

---

## Module structure

```text
training/
│
├── train_segmentation_par_patchs.py
└── train_segmentation_par_biopsie.py
```

---

## `train_segmentation_par_patchs.py`

Trains the semantic segmentation model using a patch-based
dataset.

The script loads image patches and their corresponding
ground-truth segmentation masks and uses them to train the model.

The training process includes:

- dataset loading
- preprocessing
- data augmentation
- model training
- validation
- loss monitoring
- model checkpointing

This approach is used to train the segmentation model from the
prepared patch dataset.

---

## `train_segmentation_par_biopsie.py`

Trains the semantic segmentation model using a biopsy-based
dataset organization.

The data are organized according to their biopsy of origin in
order to control the separation between training and validation
data.

This approach allows the model to be evaluated on biopsies that
are not included in the training data.

The script is particularly useful for assessing the ability of
the model to generalize to previously unseen biopsies.

---

## Model architecture

The segmentation model is based on a U-Net architecture with a
ResNet34 encoder.

The model performs semantic segmentation by assigning a tissue
class to each pixel of the input image.

The segmentation classes used during training are:

| Class | Description |
|---|---|
| Fond | Background / non-tissue regions |
| CK14 | CK14-positive tissue compartment |
| Stroma | Stromal tissue |
| Autres | Other tissue regions |

---

## Training configuration

The training parameters are defined within the training scripts
and their associated configuration.

The main parameters include:

- input patch size
- batch size
- number of epochs
- learning rate
- optimizer
- loss function
- class weights
- validation split
- early stopping parameters

The training process can be adapted according to the available
dataset and computational resources.

---

## Loss function

The segmentation model is trained using a combination of
Cross-Entropy Loss and Dice Loss.

The combined loss is used to optimize both pixel-wise class
prediction and region overlap.

Class weights are applied to account for differences in the
representation of the segmentation classes.

---

## Data augmentation

Data augmentation can be applied during training to increase the
variability of the training samples.

The augmentation strategy is applied to the image and its
corresponding segmentation mask while preserving their spatial
correspondence.

---

## Validation

Model validation is performed during training to monitor the
segmentation performance on data not used for parameter
optimization.

The training process records validation loss and segmentation
metrics to monitor model performance and identify the best model
checkpoint.

---

## Biopsy-based evaluation

The biopsy-based training script allows the dataset to be
organized according to biopsy identity.

This prevents patches originating from the same held-out biopsy
from being included in the training set when performing an
independent biopsy evaluation.

The project uses progressively expanded training sets to evaluate
model performance as additional annotated biopsies are introduced.

---

## Output

The training scripts generate trained model checkpoints that can
subsequently be used by the inference modules.

Typical output:

```text
models/
│
└── model.pth
```

Additional training outputs may include:

- training loss
- validation loss
- Dice scores
- IoU scores
- learning curves
- model checkpoints

---

## Running the training

For patch-based training:

```bash
python train_segmentation_par_patchs.py
```

For biopsy-based training:

```bash
python train_segmentation_par_biopsie.py
```

The dataset paths and training parameters must be configured
before running the scripts.

---

## Relationship with the project pipeline

The training modules are located between data preparation and
inference.

The overall workflow is:

```text
Data preparation
       │
       ▼
Prepared image / mask patches
       │
       ▼
    Training
       │
       ▼
 Trained segmentation model
       │
       ▼
    Inference
       │
       ▼
Predicted tissue compartments
       │
       ▼
    Evaluation
```

---

## Data confidentiality

The original Whole Slide Images, biopsy data, ground-truth
annotations and laboratory datasets are not included in the
repository.

Training data and trained model files are maintained outside the
public repository.

The paths used in the scripts must therefore be configured locally
using authorized project data.

---

## Dependencies

The main libraries used by the training modules include:

- Python
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- tqdm

See the main project `requirements.txt` for the complete list of
dependencies.

---

## Project context

These modules are part of the
**P16-KLHL35 AI Annotation Pipeline**.

Their purpose is to train semantic segmentation models capable of
identifying tissue compartments in histological images and to
provide trained models for subsequent Whole Slide Image inference
and automated annotation.

### Author : Hamza Graïn
