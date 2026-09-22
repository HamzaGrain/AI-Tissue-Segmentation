# Pipeline

## Overview

The P16-KLHL35 AI Annotation Pipeline is a modular workflow designed to transform manually annotated histological data into an automated tissue segmentation and QuPath annotation workflow.

The complete pipeline can be divided into five main stages:

```text
Data Preparation
       ↓
Model Training
       ↓
Whole-Slide Inference
       ↓
Segmentation Post-processing
       ↓
Polygon / QuPath Annotation
       ↓
Evaluation
````
The pipeline is designed to keep data preparation, model training, inference, annotation generation, and evaluation as separate modules.

## 1. Complete Workflow
````
                         ┌─────────────────────┐
                         │   Whole Slide Image │
                         │       (.ndpi)       │
                         └──────────┬──────────┘
                                    │
                                    │
                    ┌───────────────▼────────────────┐
                    │       DATA PREPARATION         │
                    │                                │
                    │  QuPath annotations (GeoJSON)  │
                    │              ↓                 │
                    │       Segmentation mask        │
                    │              ↓                 │
                    │     Image / mask patches       │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        MODEL TRAINING         │
                    │                               │
                    │       U-Net + ResNet34       │
                    │                               │
                    │   Cross-Entropy + Dice Loss   │
                    └───────────────┬───────────────┘
                                    │
                                    │ trained model
                                    ▼
                    ┌───────────────────────────────┐
                    │      WHOLE-SLIDE INFERENCE    │
                    │                               │
                    │  WSI → overlapping patches    │
                    │              ↓                │
                    │        model prediction        │
                    │              ↓                │
                    │     prediction reconstruction  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       POST-PROCESSING         │
                    │                               │
                    │  Confidence filtering         │
                    │  Morphological processing     │
                    │  Small-region removal         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     POLYGON GENERATION        │
                    │                               │
                    │  Segmentation mask             │
                    │          ↓                    │
                    │  Contour extraction            │
                    │          ↓                    │
                    │  Polygon processing            │
                    │          ↓                    │
                    │  GeoJSON export                │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       QuPath        │
                         │                     │
                         │ AI annotation       │
                         │ + manual correction  │
                         └─────────────────────┘
````
## 2. Data Preparation

The first stage converts manually generated QuPath annotations into a format suitable for deep learning.
````
QuPath
  │
  └── Manual annotations
          │
          ▼
       GeoJSON
          │
          ▼
  geojson_to_mask.py
          │
          ▼
 Segmentation mask
          │
          ├──────────────┐
          │              │
          ▼              ▼
      WSI image       Mask
          │              │
          └──────┬───────┘
                 ▼
       generate_patches.py
                 │
                 ▼
       Image / mask patches
````
````
Main modules
src/data_preparation/
├── geojson_to_mask.py
└── generate_patches.py
geojson_to_mask.py
````

Converts QuPath GeoJSON annotations into pixel-level segmentation masks.

The resulting mask associates each pixel with one of the four segmentation classes:
````
Fond
CK14
Stroma
Autres
generate_patches.py
````
Divides the WSI and corresponding segmentation mask into paired patches.

The project uses:
````
patch size: 512 × 512
stride: 256
overlapping patches
WSI level 4
````
The resulting image and mask patches form the training dataset.

## 3. Model Training

The generated patches are used to train the semantic segmentation model.
````
Image patches
      +
Ground-truth masks
      │
      ▼
Training dataset
      │
      ▼
┌──────────────────────┐
│      U-Net           │
│                      │
│   ResNet34 Encoder   │
│          +           │
│    U-Net Decoder     │
└──────────┬───────────┘
           │
           ▼
   Pixel-level prediction
           │
           ▼
     Loss calculation
           │
           ▼
   Model optimization
````
The training objective combines:
````
Cross-Entropy Loss
Dice Loss
````
with equal weighting.

Class weights are also used to account for differences in class representation.

The training scripts are located in:
````
src/training/
````
Main training modules:
````
train_segmentation_par_patchs.py
train_segmentation_par_biopsie.py
````
## 4. Whole-Slide Inference

Once a model has been trained, it can be applied to a complete WSI.

The WSI is not processed as a single image.

Instead, it is divided into overlapping patches using the same general patching strategy used during training.
````
Whole Slide Image
        │
        ▼
   Patch extraction
        │
        ├── Patch 1
        ├── Patch 2
        ├── Patch 3
        ├── ...
        └── Patch N
        │
        ▼
   Model prediction
        │
        ▼
 Patch-level predictions
        │
        ▼
 Prediction aggregation
        │
        ▼
 Slide-level segmentation
````
The main WSI inference script is:
````
src/inference/predict_wsi.py
````
The overlapping predictions are aggregated to reconstruct the segmentation at slide level.

## 5. Prediction Reconstruction

Because neighbouring patches overlap, several predictions may correspond to the same region of the WSI.

The inference process aggregates the predictions from overlapping patches to obtain a single slide-level segmentation.

Conceptually:
````
Patch A ──────────┐
                  │
Patch B ──────────┼──→ Prediction aggregation
                  │
Patch C ──────────┘
                         │
                         ▼
                 Slide-level prediction
````
This reduces discontinuities between individual patch predictions and allows the model to produce a continuous segmentation over the processed WSI region.

## 6. Post-processing

The raw segmentation is processed before conversion into polygon annotations.

The post-processing stage includes:
````
Raw segmentation
       │
       ▼
Confidence filtering
       │
       ▼
Morphological processing
       │
       ▼
Small-region removal
       │
       ▼
Cleaned segmentation
````
The purpose is to remove small isolated regions and reduce prediction artefacts before generating polygon geometries.

The post-processing implementation is located in:
````
src/inference/predict_geojson_to_QuPath/
````
## 7. Polygon Extraction

The cleaned segmentation mask is converted into vector geometries.

Each tissue class is processed separately.
````
Segmentation mask
       │
       ▼
Class-specific binary masks
       │
       ▼
Contour extraction
       │
       ▼
Polygon generation
       │
       ▼
Polygon validation / processing
````
The polygon extraction modules are:
````
polygon_extraction.py
polygon_processing.py
````
The resulting geometries represent the predicted tissue compartments.

## 8. GeoJSON Export

The processed polygons are converted into GeoJSON annotations.
````
Predicted polygons
        │
        ▼
Class information
        │
        ▼
Coordinate transformation
        │
        ▼
GeoJSON FeatureCollection
````
The export is implemented in:
````
geojson_export.py
````
The resulting GeoJSON can then be imported into QuPath.

## 9. QuPath Integration

The final output of the automated pipeline is designed to be compatible with QuPath.
````
AI prediction
      │
      ▼
Segmentation mask
      │
      ▼
Polygon extraction
      │
      ▼
GeoJSON
      │
      ▼
QuPath
      │
      ▼
AI-generated annotations
      │
      ▼
Manual review / correction
````
The generated annotations are intended to serve as a first annotation proposal.

The user can inspect and correct the generated regions in QuPath before using them for downstream quantitative analysis.

## 10. Quality Assurance

Additional quality-control information can be generated during the annotation pipeline.

The QA stage can be used to identify potentially problematic regions in the generated segmentation.

The corresponding module is:
````
quality_assurance.py
````
The QA step complements the segmentation and polygon generation stages rather than replacing manual review.

## 11. Evaluation Pipeline

The generated annotations can be evaluated against manual reference annotations.

The evaluation workflow is:
````
Manual annotations
        │
        ├─────────────────────────┐
        │                         │
        ▼                         ▼
Quantitative analysis       Spatial analysis
        │                         │
        ▼                         ▼
Compartment proportions     Polygon overlap
        │                         │
        ▼                         ▼
Quantitative concordance    Dice / IoU
                             Precision / Recall
````
Additional evaluations were performed to assess:
````
patch-level model performance;
independent biopsy performance;
quantitative concordance;
spatial concordance;
intra-annotator reproducibility;
annotation time savings.
````
The corresponding scripts are located in:
````
src/evaluation/
````
## 12. End-to-End Architecture

The complete software architecture can be summarized as:
````
                    DATA
                     │
                     ▼
        ┌────────────────────────┐
        │   Data Preparation     │
        │                        │
        │ GeoJSON → Mask         │
        │ WSI → Patches          │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │     Model Training     │
        │                        │
        │ U-Net + ResNet34       │
        │ CE + Dice Loss         │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │       Inference        │
        │                        │
        │ WSI → Patches          │
        │ Patches → Predictions  │
        │ Predictions → Mask     │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │    Post-processing     │
        │                        │
        │ Filtering              │
        │ Morphology             │
        │ Region cleaning        │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  Polygon Generation    │
        │                        │
        │ Mask → Contours        │
        │ Contours → Polygons    │
        │ Polygons → GeoJSON     │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │        QuPath          │
        │                        │
        │ AI annotations         │
        │ Manual review          │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │      Evaluation        │
        │                        │
        │ Dice / IoU              │
        │ Quantitative agreement │
        │ Spatial agreement      │
        │ Time savings           │
        └────────────────────────┘
````
## 13. Repository Mapping

The main pipeline stages correspond to the following repository modules:
````
src/
│
├── data_preparation/
│   ├── geojson_to_mask.py
│   └── generate_patches.py
│
├── training/
│   ├── train_segmentation_par_patchs.py
│   └── train_segmentation_par_biopsie.py
│
├── inference/
│   ├── predict_wsi.py
│   │
│   └── predict_geojson_to_QuPath/
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
    ├── compare_geojson_manual_IA.py
    ├── compare_distribution.py
    ├── distribution_compartiments.py
    ├── verification_patches.py
    └── evaluation_model.py
````
## 14. Design Principles

The pipeline was developed according to several principles:

### Modularity

Each major stage is implemented as a separate module.

This makes it possible to modify data preparation, training, inference, post-processing, or evaluation independently.

### Reproducibility

Training and evaluation procedures are separated from the original biological data.

The repository documents the computational workflow without distributing sensitive research data.

### Scalability

The use of patches allows the model to process whole-slide images that are too large to fit directly into GPU memory.

### Human-in-the-loop annotation

The system is designed to generate an initial annotation that can be reviewed and corrected by the user in QuPath.

### Quantitative evaluation

The pipeline is evaluated not only through segmentation metrics but also through global tissue proportions, spatial agreement, manual reproducibility, and annotation time.

## 15. Data Confidentiality

The complete pipeline was developed using research data that are not distributed through this repository.

The repository does not contain:
````
original whole-slide images;
patient-related information;
biological images;
manual annotation files;
original GeoJSON annotations;
segmentation masks derived from the biological samples;
trained model weights.
````
The code and documentation are provided to demonstrate the computational architecture and methodology of the project without exposing the underlying research data.
