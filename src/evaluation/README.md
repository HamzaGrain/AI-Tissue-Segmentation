# Evaluation

## Overview

This directory contains the scripts used to evaluate the quality
of the segmentation dataset, the trained AI model, and the
agreement between manual and AI-generated tissue compartment
annotations.

The evaluation is performed at several levels, from verification
of the generated patches to quantitative and spatial comparison
of the final annotations.

---

## Evaluation workflow

The evaluation process includes the following steps:

```text
Dataset patches
      │
      ▼
Patch verification
      │
      ▼
Model evaluation
      │
      ▼
Tissue compartment distribution
      │
      ▼
Manual vs AI distribution comparison
      │
      ▼
Manual vs AI spatial comparison
```

---

## Module structure

```text
evaluation/
│
├── compare_geojson_manual_IA.py
├── compare_distribution.py
├── distribution_compartiments.py
├── verification_patches.py
└── evaluation_model.py
```

---

## `verification_patches.py`

Verifies the image and segmentation-mask patches generated during
dataset preparation.

The module checks the correspondence between image patches and
their associated ground-truth masks.

This step helps identify potential inconsistencies in the dataset
before model training.

---

## `evaluation_model.py`

Evaluates the performance of the trained semantic segmentation
model.

The module computes segmentation performance metrics for the
different tissue classes and summarizes the overall model
performance.

The evaluation can be used to assess the quality of the predicted
segmentation masks on the validation data.

---

## `distribution_compartiments.py`

Analyzes the distribution of tissue compartments within annotated
biopsies.

The module calculates the proportion of each tissue class and
provides a quantitative description of the compartment
composition.

The results can be used to characterize the distribution of
tissue compartments within the samples.

---

## `compare_distribution.py`

Compares tissue compartment distributions between manual and
AI-generated annotations.

The module evaluates the quantitative agreement between the
proportions obtained from the manual reference annotations and
those obtained from the AI predictions.

This provides a whole-biopsy comparison of the predicted tissue
composition.

---

## `compare_geojson_manual_IA.py`

Compares manual and AI-generated GeoJSON annotations at the
spatial level.

The module evaluates the spatial agreement between the manual
reference annotations and the AI predictions for each tissue
compartment.

The comparison is based on the geometric overlap between the
manual and predicted regions.

The evaluation includes metrics such as:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall

The metrics can be calculated separately for each tissue class
and summarized using macro-averaged values.

---

## Evaluation levels

The scripts provide complementary evaluation levels.

### Dataset verification

`verification_patches.py`

Checks the integrity and correspondence of the image and
ground-truth mask patches used for training.

### Model performance

`evaluation_model.py`

Evaluates the segmentation performance of the trained model
using segmentation metrics.

### Tissue distribution

`distribution_compartiments.py`

Characterizes the proportion of the different tissue
compartments within the annotated biopsies.

### Quantitative agreement

`compare_distribution.py`

Compares the global tissue compartment proportions obtained
from manual and AI-generated annotations.

### Spatial agreement

`compare_geojson_manual_IA.py`

Evaluates how closely the spatial organization of the AI-generated
annotations matches the manual reference annotations.

---

## Evaluation outputs

Depending on the script and configuration, the evaluation modules
can generate:

- segmentation metrics
- tissue compartment proportions
- manual vs AI quantitative differences
- spatial overlap metrics
- comparison tables
- evaluation reports

These outputs are used to assess both the quantitative agreement
and the spatial correspondence between manual and AI-generated
annotations.

---

## Data

The evaluation scripts operate on project-specific histological
images, segmentation masks and annotation files.

The original biological data, Whole Slide Images, ground-truth
annotations and confidential experimental results are not included
in this repository.

The paths required by the scripts must therefore be configured
locally.

---

## Dependencies

The evaluation modules mainly rely on:

- Python
- NumPy
- OpenCV
- Shapely
- GeoPandas, when applicable
- pandas
- scikit-learn, when applicable

See the main project `requirements.txt` for the complete list of
dependencies.

---

## Project context

These evaluation modules are part of the
**P16-KLHL35 AI Annotation Pipeline**.

Their purpose is to provide complementary measurements of model
performance and annotation agreement, supporting the evaluation of
automated tissue compartment annotation for subsequent
quantitative analysis.

### Author : Hamza Graïn
