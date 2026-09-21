# Source Code

## Overview

This directory contains the source code developed for the
P16-KLHL35 AI Annotation Pipeline.

The source code is organized into modular components covering the
complete workflow, from data preparation to model training,
Whole Slide Image inference and evaluation.

The main objective is to provide a modular and reproducible
software structure for the automated segmentation and annotation
of histological tissue compartments.

---

## Architecture

The source code is organized into four main stages:

```text
Data preparation
       │
       ▼
    Training
       │
       ▼
    Inference
       │
       ▼
    Evaluation
```

Each stage is implemented as a separate module.

---

## Module structure

```text
src/
│
├── data_preparation/
│   ├── README.md
│   ├── geojson_to_mask.py
│   └── generate_patches.py
│
├── training/
│   ├── README.md
│   ├── train_segmentation_par_patchs.py
│   └── train_segmentation_par_biopsie.py
│
├── inference/
│   ├── README.md
│   ├── predict_wsi.py
│   │
│   └── predict_geojson_to_QuPath/
│       ├── README.md
│       ├── config.py
│       ├── model.py
│       ├── inference.py
│       ├── postprocessing.py
│       ├── polygon_extraction.py
│       ├── polygon_processing.py
│       ├── geojson_export.py
│       ├── overlay.py
│       ├── statistics.py
│       ├── quality_assurance.py
│       └── predict_geojson.py
│
└── evaluation/
    ├── README.md
    ├── verification_patches.py
    ├── evaluation_model.py
    ├── distribution_compartiments.py
    ├── compare_distribution.py
    └── compare_geojson_manual_IA.py
```

---

## `data_preparation/`

Contains the modules responsible for preparing the training
dataset.

The main steps are:

- conversion of QuPath GeoJSON annotations into segmentation masks
- extraction of image and mask patches
- preparation of paired image / ground-truth data

See [`data_preparation/README.md`](data_preparation/README.md) for
more information.

---

## `training/`

Contains the scripts used to train the semantic segmentation model.

Two training approaches are provided:

- patch-based training
- biopsy-based training

The trained models are subsequently used during inference.

See [`training/README.md`](training/README.md) for more information.

---

## `inference/`

Contains the modules used to apply the trained model to new
Whole Slide Images.

The inference stage includes:

- WSI processing
- patch-based prediction
- segmentation reconstruction
- post-processing
- polygon extraction
- GeoJSON generation
- QuPath-compatible annotation export

See [`inference/README.md`](inference/README.md) for more
information.

---

## `evaluation/`

Contains the scripts used to evaluate the dataset, trained model
and generated annotations.

The evaluation covers:

- patch verification
- model segmentation performance
- tissue compartment distribution
- quantitative agreement between manual and AI annotations
- spatial agreement between manual and AI annotations

See [`evaluation/README.md`](evaluation/README.md) for more
information.

---

## Complete pipeline

The complete source-code workflow can be summarized as:

```text
                    ┌─────────────────────┐
                    │   Histological WSI  │
                    │       + QuPath      │
                    │     annotations     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Preparation   │
                    │                     │
                    │ GeoJSON → Mask      │
                    │ WSI → Patches       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Training       │
                    │                     │
                    │ U-Net + ResNet34    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Inference       │
                    │                     │
                    │ WSI → Segmentation  │
                    │ → Polygons →        │
                    │ GeoJSON → QuPath    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Evaluation      │
                    │                     │
                    │ Model + Quantitative │
                    │ + Spatial analysis  │
                    └─────────────────────┘
```

---

## Design principles

The source code follows a modular organization in order to
separate the different stages of the machine learning pipeline.

This organization makes it possible to:

- prepare data independently from model training
- train and evaluate different model configurations
- perform inference on new Whole Slide Images
- separate segmentation from polygon processing
- evaluate predictions using complementary metrics
- maintain and extend individual components of the pipeline

---

## Technologies

The source code relies mainly on:

- Python
- PyTorch
- OpenCV
- NumPy
- OpenSlide
- Shapely
- GeoJSON
- QuPath

Additional dependencies are listed in the main project
`requirements.txt`.

---

## Data and confidentiality

The source code is separated from the biological data used during
the project.

The repository does not contain:

- Whole Slide Images
- original biopsy images
- ground-truth annotations
- laboratory datasets
- confidential experimental results

The required data paths must be configured locally using
authorized project data.

---

## Project context

The `src/` directory contains the implementation of the
**P16-KLHL35 AI Annotation Pipeline**, developed during a research
project focused on the automated segmentation and annotation of
histological tissue compartments.

The pipeline is designed to support annotation in QuPath and
facilitate subsequent quantitative analysis of tissue biomarkers.

### Author : Hamza Graïn
