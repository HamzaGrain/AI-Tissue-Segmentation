# P16-KLHL35 AI Annotation Pipeline

## Overview

This project focuses on the automated segmentation and annotation
of histological tissue compartments using deep learning.

The objective is to assist the manual annotation workflow used in
digital pathology and facilitate the subsequent quantitative
analysis of tissue biomarkers.

The pipeline processes histological Whole Slide Images (WSI),
generates semantic segmentation predictions, converts the predicted
regions into polygon annotations, and exports them in a format
compatible with QuPath.

The project was developed during a research internship in the
context of the analysis of anorectal junction biopsy samples.

---

## Objective

Manual annotation of histological tissue compartments is a
time-consuming process, particularly when working with
high-resolution Whole Slide Images.

This project aims to develop an automated segmentation workflow
capable of:

- identifying the main tissue compartments
- generating pixel-level segmentation predictions
- converting segmentation results into polygon annotations
- exporting annotations for use in QuPath
- assisting the manual annotation process
- facilitating quantitative analysis of tissue compartments

The generated annotations are intended to assist the annotation
workflow rather than replace expert validation.

---

## Tissue compartments

The semantic segmentation model uses four classes:

| Class | Description |
|---|---|
| Fond | Background / non-tissue regions |
| CK14 | CK14-positive tissue compartment |
| Stroma | Stromal tissue |
| Autres | Other tissue regions |

The `Fond` class is used during semantic segmentation, while the
main tissue compartments are subsequently processed for polygon
generation and quantitative analysis.

---

## Pipeline

The complete workflow is:

```text
                     Whole Slide Image
                           (.ndpi)
                              │
                              ▼
                    Data Preparation
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              QuPath GeoJSON        WSI patches
                    │                   │
                    ▼                   │
             Segmentation mask         │
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Training Dataset
                              │
                              ▼
                         Model Training
                              │
                              ▼
                       Trained U-Net
                       + ResNet34
                              │
                              ▼
                           Inference
                              │
                              ▼
                     Segmentation Mask
                              │
                              ▼
                       Post-processing
                              │
                              ▼
                      Polygon Extraction
                              │
                              ▼
                     Polygon Processing
                              │
                              ▼
                       GeoJSON Export
                              │
                              ▼
                            QuPath
                              │
                              ▼
                          Evaluation
```

---

## Model

The segmentation model is based on a **U-Net architecture with a
ResNet34 encoder**.

The model performs semantic segmentation by assigning a tissue
class to each pixel of an input image patch.

The training objective combines:

- Cross-Entropy Loss
- Dice Loss

Class weighting is used to account for differences in the
representation of the segmentation classes.

---

## Dataset

The project uses histological biopsy samples acquired as
Whole Slide Images.

The complete dataset contains:

- 40 biopsies
- 20 P16 samples
- 20 KLHL35 samples
- 20 paired serial-section samples

Manual annotations were performed on a subset of the biopsies to
create the training and evaluation data.

The annotated data were converted into segmentation masks and
subsequently divided into image / mask patches for model training.

The project dataset contains several thousand image patches
generated from manually annotated biopsies.

---

## Data preparation

The data preparation stage converts manually generated QuPath
annotations into pixel-level segmentation masks.

The masks are then aligned with the corresponding Whole Slide
Images and divided into patches.

Main steps:

```text
QuPath annotations
        │
        ▼
    GeoJSON
        │
        ▼
Segmentation mask
        │
        ▼
Patch extraction
        │
        ▼
Image / mask pairs
```

See [`src/data_preparation/README.md`](src/data_preparation/README.md)
for details.

---

## Training

Two training approaches are implemented:

### Patch-based training

The model is trained using image patches and their corresponding
ground-truth segmentation masks.

### Biopsy-based training

The data are organized according to biopsy identity to allow
evaluation on previously unseen biopsies.

This approach is used to assess the ability of the model to
generalize beyond the biopsies used during training.

See [`src/training/README.md`](src/training/README.md) for details.

---

## Inference

The trained model is applied to new Whole Slide Images using a
patch-based inference strategy.

The WSI is divided into overlapping patches and each patch is
processed independently by the trained model.

The resulting predictions are then aggregated to reconstruct the
segmentation at slide level.

```text
WSI
 │
 ▼
Overlapping patches
 │
 ▼
Model predictions
 │
 ▼
Probability aggregation
 │
 ▼
Slide-level segmentation
```

See [`src/inference/README.md`](src/inference/README.md) for details.

---

## QuPath annotation generation

The predicted segmentation is subsequently converted into polygon
geometries.

The annotation pipeline performs:

1. post-processing of the segmentation mask
2. polygon extraction
3. polygon validation and processing
4. GeoJSON generation
5. visualization and quality checks

The resulting GeoJSON file can be imported into QuPath.

```text
Segmentation mask
        │
        ▼
Post-processing
        │
        ▼
Polygon extraction
        │
        ▼
Polygon processing
        │
        ▼
GeoJSON
        │
        ▼
QuPath
```

The main tissue classes exported as annotations are:

- CK14
- Stroma
- Autres

The generated annotations are intended to provide an initial
annotation proposal that can subsequently be reviewed and corrected
in QuPath.

---

## Evaluation

The project includes several complementary evaluation approaches.

### Model performance

The segmentation model is evaluated using metrics such as:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall

### Quantitative agreement

Manual and AI-generated annotations are compared at whole-biopsy
level by measuring the proportion of each tissue compartment.

This evaluates whether the automated segmentation reproduces the
global tissue composition of the manual reference.

### Spatial agreement

The spatial correspondence between manual and AI-generated
annotations is evaluated using geometric overlap metrics.

### Dataset verification

Image and segmentation-mask patches are also verified to ensure
their correspondence before model training.

See [`src/evaluation/README.md`](src/evaluation/README.md) for details.

---

## Repository structure

```text
P16-KLHL35-AI-Annotation-Pipeline/
│
├── README.md
├── NOTICE.md
├── requirements.txt
├── .gitignore
│
├── src/
│   │
│   ├── README.md
│   │
│   ├── data_preparation/
│   │   ├── README.md
│   │   ├── geojson_to_mask.py
│   │   └── generate_patches.py
│   │
│   ├── training/
│   │   ├── README.md
│   │   ├── train_segmentation_par_patchs.py
│   │   └── train_segmentation_par_biopsie.py
│   │
│   ├── inference/
│   │   ├── README.md
│   │   ├── predict_wsi.py
│   │   │
│   │   └── predict_geojson_to_QuPath/
│   │       ├── README.md
│   │       ├── config.py
│   │       ├── model.py
│   │       ├── inference.py
│   │       ├── postprocessing.py
│   │       ├── polygon_extraction.py
│   │       ├── polygon_processing.py
│   │       ├── geojson_export.py
│   │       ├── overlay.py
│   │       ├── statistics.py
│   │       ├── quality_assurance.py
│   │       └── predict_geojson.py
│   │
│   └── evaluation/
│       ├── README.md
│       ├── verification_patches.py
│       ├── evaluation_model.py
│       ├── distribution_compartiments.py
│       ├── compare_distribution.py
│       └── compare_geojson_manual_IA.py
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   └── README.md
│
└── docs/
    ├── methodology.md
    └── architecture.png
```

---

## Technologies

The project was developed primarily using:

- Python
- PyTorch
- OpenCV
- NumPy
- OpenSlide
- Shapely
- GeoJSON
- QuPath

The complete list of Python dependencies is provided in
`requirements.txt`.

---

## Main technical components

| Component | Role |
|---|---|
| U-Net | Semantic segmentation architecture |
| ResNet34 | Encoder / feature extraction |
| PyTorch | Deep learning framework |
| OpenSlide | Whole Slide Image reading |
| OpenCV | Image and mask processing |
| Shapely | Polygon and geometric processing |
| GeoJSON | Annotation exchange format |
| QuPath | Digital pathology visualization and annotation |

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd P16-KLHL35-AI-Annotation-Pipeline
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The repository is designed to contain the source code and
configuration examples, while project-specific biological data
remain outside the repository.

---

## Usage

The general workflow is:

### 1. Prepare the dataset

Convert QuPath annotations into segmentation masks and generate
image / mask patches.

### 2. Train the model

Run one of the training scripts according to the desired dataset
organization.

### 3. Run inference

Apply the trained model to a Whole Slide Image.

### 4. Generate QuPath annotations

Convert the predicted segmentation into polygon annotations and
export them as GeoJSON.

### 5. Evaluate the results

Compare the automated predictions with the manual reference
annotations using quantitative and spatial metrics.

---

## Data confidentiality

This repository does **not** contain the biological data used
during the project.

The following materials are excluded:

- Whole Slide Images
- original biopsy images
- raw histological data
- ground-truth annotations
- laboratory datasets
- confidential experimental results

The required data must be stored locally and accessed using
authorized paths.

The repository is intended to demonstrate the software
architecture, machine learning workflow and evaluation methodology
without distributing the underlying biological data.

---

## Intellectual Property

This project was developed during a research internship.

The source code was developed specifically for this project.

The biological data, histological images, annotations and related
research data remain subject to the rights and conditions of the
host laboratory and the applicable internship or institutional
agreements.

This repository is private and access is restricted to authorized
users.

No redistribution, publication or reuse of confidential project
data is authorized without appropriate permission from the
relevant rights holders.

See [`NOTICE.md`](NOTICE.md) for additional information.

---

## Project status

The pipeline includes the main stages required for automated
histological tissue compartment annotation:

- dataset preparation
- semantic segmentation model training
- Whole Slide Image inference
- polygon generation
- GeoJSON export
- QuPath integration
- quantitative evaluation
- spatial evaluation

The generated annotations are intended as an assistance tool for
the annotation workflow and remain subject to expert review.

---

## Author

**Hamza Graïn**

Developed as part of a research internship focused on
deep learning and automated annotation in digital pathology.
