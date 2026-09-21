# Predict GeoJSON to QuPath

## Overview

This module provides an automated pipeline for converting semantic
segmentation predictions from histological Whole Slide Images (WSI)
into polygon annotations compatible with QuPath.

The pipeline is designed to automatically identify the main tissue
compartments of histological biopsies and generate corresponding
QuPath annotations.

The predicted tissue classes are:

- **CK14**
- **Stroma**
- **Autres**

The module processes the segmentation prediction through several
steps, from model inference to polygon generation and GeoJSON export.

---

## Pipeline

The complete workflow is:

```
Whole Slide Image (.ndpi)
          │
          ▼
      Model loading
          │
          ▼
    WSI inference
          │
          ▼
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


## Module structure

predict_geojson_to_QuPath/
│
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
`
## config.py

Centralized configuration of the pipeline.

It contains:

input and output paths
WSI resolution level
patch size and stride
model parameters
post-processing parameters
tissue classes
visualization colors
export options
quality assurance thresholds


## model.py

Handles the loading of the trained segmentation model and the
Whole Slide Image.


## inference.py

Performs inference on the Whole Slide Image.

The WSI is processed using overlapping image patches. The model
generates class probabilities for each patch, which are then
combined to reconstruct the segmentation prediction at slide level.

The module returns:

the predicted segmentation mask
the confidence map
the class probability maps


## postprocessing.py

Cleans the raw segmentation prediction.

The processing may include:

confidence filtering
morphological operations
removal of small regions
segmentation mask refinement

The objective is to obtain a cleaner segmentation mask before
converting it into polygon geometries.


## polygon_extraction.py

Converts the segmentation mask into polygon geometries.

For each tissue class, the module:

extracts contours from the segmentation mask
identifies external contours and holes
creates Shapely polygon geometries
validates invalid geometries when necessary


## polygon_processing.py

Processes the extracted polygon geometries.

The module performs:

polygon validation
removal of very small regions
polygon simplification
merging of nearby polygons
computation of polygon statistics


## geojson_export.py

Converts the processed polygon geometries into a GeoJSON file
compatible with QuPath.

Each annotation contains its corresponding tissue classification
and visualization color.

Coordinates are converted back to the original WSI coordinate
system before export.


## overlay.py

Generates visualization outputs for inspecting the segmentation
results.

The module can generate:

predicted segmentation masks
color overlays
segmentation contours
confidence maps
class legends

These visualizations can be used to visually inspect the prediction
before importing the annotations into QuPath.


## statistics.py

Computes statistics associated with the generated annotations,
including geometric and class-related information.


## quality_assurance.py

Performs additional quality checks on the generated annotations.

It identifies annotations that may require visual verification
after automated prediction.


## predict_geojson.py

Main entry point of the pipeline.

It coordinates all processing steps:

Model loading
      ↓
WSI loading
      ↓
Inference
      ↓
Post-processing
      ↓
Polygon extraction
      ↓
Polygon processing
      ↓
GeoJSON export
      ↓
Visualization
      ↓
Statistics
      ↓
Quality assurance


## Configuration

Before running the pipeline, modify the paths in config.py.

Example:

NDPI_PATH = r"path/to/your/slide.ndpi"

MODEL_PATH = r"path/to/your/model.pth"

OUTPUT_DIR = "results"

The remaining parameters control the inference resolution,
patch extraction, model configuration, post-processing and
export settings.

## Input

The pipeline requires:

1. Whole Slide Image

Supported input:

.ndpi

The WSI is read using OpenSlide.


2. Trained segmentation model

A trained PyTorch model is required.

Example:


## model.pth

The model must correspond to the segmentation architecture and
number of classes defined in config.py.


## Output

The pipeline generates several outputs in the configured
output directory.


Typical outputs include:

results/
│
├── prediction_mask.png
├── prediction_clean.png
├── prediction_overlay.png
├── prediction_contours.png
├── confidence_map.png
├── legend.png
├── prediction.geojson
├── annotation_statistics.csv
├── annotation_report.txt
└── annotation_warnings.csv


The main output for QuPath is:

prediction.geojson

This file contains the automatically generated polygon annotations.

Running the pipeline

After configuring the input paths, run:

python predict_geojson.py

The pipeline will automatically execute the complete workflow.

## QuPath

The generated GeoJSON can be imported into QuPath to visualize
the automatically generated tissue compartment annotations.


The annotations are classified according to the following classes:

Class ID	Class
1	CK14
2	Stroma
3	Autres

The generated annotations are intended to assist the manual
annotation process and facilitate the identification and
quantification of tissue compartments.

## Important

The repository does not include:

Whole Slide Images
Original biopsy images
Ground-truth annotations
Laboratory datasets
Confidential experimental results

These data remain external to the repository.

The paths shown in the configuration files are therefore examples
and must be replaced by the user with their own local paths.


## Dependencies

Main Python libraries used by this module include:

Python
PyTorch
OpenCV
NumPy
OpenSlide
Shapely
tqdm

See the main project requirements.txt for the complete list of
dependencies.

## Project context

This module was developed as part of a research project focused on
the automated annotation of histological tissue compartments using
deep learning.

The objective is to assist the annotation workflow in QuPath and
facilitate subsequent quantitative analysis of tissue biomarkers.

### Author: Hamza Graïn
