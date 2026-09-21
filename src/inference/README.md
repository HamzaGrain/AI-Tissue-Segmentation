# Inference

## Overview

This directory contains the modules used to apply the trained
semantic segmentation model to new Whole Slide Images (WSI).

The inference workflow is divided into two main stages:

1. Whole Slide Image segmentation
2. Conversion of the segmentation result into polygon annotations
   compatible with QuPath

The complete workflow transforms a WSI into automatically generated
tissue compartment annotations.

---

## Inference workflow

```text
Whole Slide Image (.ndpi)
          │
          ▼
      WSI loading
          │
          ▼
     Patch extraction
          │
          ▼
   Model prediction
          │
          ▼
 Probability aggregation
          │
          ▼
 Segmentation reconstruction
          │
          ▼
 Predicted segmentation
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
    GeoJSON export
          │
          ▼
        QuPath
```

---

## Module structure

```text
inference/
│
├── predict_wsi.py
│
└── predict_geojson_to_QuPath/
    ├── config.py
    ├── model.py
    ├── inference.py
    ├── postprocessing.py
    ├── polygon_extraction.py
    ├── polygon_processing.py
    ├── geojson_export.py
    ├── overlay.py
    ├── statistics.py
    ├── quality_assurance.py
    └── predict_geojson.py
```

---

## `predict_wsi.py`

Performs semantic segmentation inference on a Whole Slide Image
using a previously trained PyTorch model.

The WSI is divided into overlapping patches. Each patch is processed
by the trained model to obtain class probabilities.

The predictions from overlapping patches are then aggregated to
reconstruct the segmentation result at slide level.

The module produces the predicted segmentation mask and associated
prediction information.

---

## `predict_geojson_to_QuPath`

This module extends the inference workflow from segmentation masks
to QuPath-compatible polygon annotations.

It processes the segmentation prediction through several stages:

```text
Segmentation prediction
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
    GeoJSON generation
          │
          ▼
        QuPath
```

The module contains the following components:

### `config.py`

Centralized configuration of the prediction and annotation
pipeline.

### `model.py`

Handles the loading of the trained segmentation model and the
Whole Slide Image.

### `inference.py`

Performs patch-based inference and combines predictions from
overlapping regions.

### `postprocessing.py`

Refines the raw segmentation prediction before polygon extraction.

### `polygon_extraction.py`

Converts segmentation regions into polygon geometries.

### `polygon_processing.py`

Processes and validates the extracted polygon geometries.

### `geojson_export.py`

Converts the processed geometries into a GeoJSON file compatible
with QuPath.

### `overlay.py`

Generates visualization outputs for inspecting the segmentation
results.

### `statistics.py`

Computes statistics associated with the generated annotations.

### `quality_assurance.py`

Performs additional checks to identify annotations that may require
visual verification.

### `predict_geojson.py`

Main entry point coordinating the complete segmentation-to-GeoJSON
workflow.

---

## Configuration

The input WSI, trained model and output paths must be configured
before running the inference scripts.

Example:

```python
NDPI_PATH = r"path/to/your/slide.ndpi"

MODEL_PATH = r"path/to/your/model.pth"

OUTPUT_DIR = "results"
```

Additional parameters control the WSI resolution level, patch
size, stride, model configuration and post-processing parameters.

---

## Input

The inference pipeline requires:

### 1. Whole Slide Image

Supported input:

```text
.ndpi
```

The WSI is read using OpenSlide.

### 2. Trained segmentation model

A trained PyTorch model compatible with the segmentation
architecture is required.

Example:

```text
model.pth
```

---

## Output

The first stage of the inference pipeline produces a slide-level
segmentation prediction.

The complete annotation pipeline can additionally generate:

```text
results/
│
├── prediction_mask.png
├── prediction_clean.png
├── prediction_overlay.png
├── prediction_contours.png
├── confidence_map.png
├── prediction.geojson
├── annotation_statistics.csv
├── annotation_report.txt
└── annotation_warnings.csv
```

The main output for QuPath is:

```text
prediction.geojson
```

---

## Running the inference

For Whole Slide Image segmentation, run:

```bash
python predict_wsi.py
```

For the complete segmentation-to-QuPath workflow, run the main
script located in:

```text
predict_geojson_to_QuPath/predict_geojson.py
```

---

## Relationship with the rest of the project

The inference modules are located between model training and
evaluation.

The overall project workflow is:

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
Segmentation / GeoJSON
       │
       ▼
    Evaluation
```

The inference stage applies the trained model to new Whole Slide
Images and prepares the resulting annotations for visualization
and quantitative analysis in QuPath.

---

## Data confidentiality

Whole Slide Images, biopsy data, ground-truth annotations, trained
model files and confidential experimental results are not included
in the repository.

The paths shown in the configuration files are examples and must
be replaced with authorized local paths.

---

## Dependencies

The main libraries used by the inference modules include:

- Python
- PyTorch
- OpenSlide
- NumPy
- OpenCV
- Shapely

See the main project `requirements.txt` for the complete list of
dependencies.

---

## Project context

These modules are part of the
**P16-KLHL35 AI Annotation Pipeline**.

Their purpose is to apply the trained semantic segmentation model
to histological Whole Slide Images and convert the resulting
predictions into structured annotations that can be inspected,
corrected and quantified in QuPath.

### Author : Hamza Graïn
