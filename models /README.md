# Models

This directory documents the deep learning models developed and used for the P16-KLHL35 AI Annotation Pipeline.

## Model Architecture

The project uses a semantic segmentation model based on:

- **U-Net** architecture
- **ResNet34** encoder
- **Cross-Entropy Loss + Dice Loss**
- Class weighting to account for class imbalance

The model performs pixel-level segmentation of histological tissue compartments.

The four segmentation classes are:

- `Fond`
- `CK14`
- `Stroma`
- `Autres`

## Model Versions

Several model versions were trained using progressively larger sets of manually annotated biopsies.

| Model | Training biopsies | Excluded biopsies |
|---|---|---|
| V1 | 001, 002 | 003, 004, 005, 006, 007, 008 |
| V2 | 001, 002, 003, 006 | 004, 005, 007, 008 |
| V3 | 001, 002, 003, 006, 007, 008 | 004, 005 |
| V4 | 001, 002, 003, 004, 006, 007, 008 | 005 |

The excluded biopsies were kept separate from the corresponding training datasets for independent evaluation.

## Model Usage

The trained models are used during the inference stage to generate semantic segmentation predictions from whole-slide images.

The general workflow is:

```text
Whole Slide Image
       ↓
Patch extraction
       ↓
Model inference
       ↓
Prediction reconstruction
       ↓
Post-processing
       ↓
Polygon extraction
       ↓
GeoJSON
       ↓
QuPath
````
The inference pipeline is documented in:
````
src/inference/README.md
````
## Model Weights

The trained model weights are not included in this repository.

They remain stored in the local project environment and are not distributed through GitHub.

This choice avoids unnecessarily distributing large model files and keeps the repository focused on the source code, software architecture, and methodology.

## Confidentiality

The histological images, whole-slide images, manual annotations, biological data, and trained model weights are not distributed through this repository.

The repository contains the software and documentation necessary to demonstrate the architecture and methodology of the project without exposing the underlying research data.
