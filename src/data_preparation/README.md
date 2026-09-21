# Data Preparation

## Overview

This directory contains the scripts used to prepare the histological
data before training the semantic segmentation models.

The data preparation pipeline converts QuPath annotations into
segmentation masks and generates image and mask patches suitable
for model training.

The preparation workflow is:

```text
Whole Slide Image (.ndpi)
          │
          ├───────────────┐
          │               │
          ▼               ▼
 QuPath GeoJSON       WSI image
          │               │
          ▼               │
   Segmentation mask     │
          │               │
          └───────┬───────┘
                  ▼
            Patch extraction
                  │
                  ▼
        Image / mask patch pairs
                  │
                  ▼
             AI dataset
```

---

## Module structure

```text
data_preparation/
│
├── geojson_to_mask.py
└── generate_patches.py
```

---

## `geojson_to_mask.py`

Converts QuPath GeoJSON annotations into pixel-level segmentation
masks.

The module reads the polygon annotations associated with a biopsy
and rasterizes them into a segmentation mask.

Each tissue compartment is represented by a specific class label
defined in the project configuration.

The generated masks are used as ground-truth segmentation targets
during model training.

---

## `generate_patches.py`

Generates image and segmentation-mask patches from Whole Slide
Images and their corresponding ground-truth masks.

The WSI is processed using a predefined patch size and stride.
For each extracted image patch, the corresponding region of the
segmentation mask is extracted at the same position.

The resulting image and mask pairs form the dataset used for
training and validation of the segmentation models.

---

## Data preparation workflow

The complete preparation process is:

### 1. Annotation conversion

QuPath polygon annotations are converted into pixel-level
segmentation masks using `geojson_to_mask.py`.

### 2. Patch generation

The Whole Slide Image and its corresponding segmentation mask are
divided into patches using `generate_patches.py`.

### 3. Image / mask pairing

Each image patch is associated with the corresponding ground-truth
mask patch.

The spatial correspondence between the image and mask is preserved
throughout the process.

### 4. Dataset organization

The generated patches are organized into image and mask datasets
that can subsequently be used by the training modules.

---

## Configuration

The paths to the input data and output directories must be
configured locally in the corresponding scripts.

Example:

```python
NDPI_PATH = r"path/to/your/slide.ndpi"

GEOJSON_PATH = r"path/to/your/annotations.geojson"

OUTPUT_MASK = r"path/to/your/output_mask.png"
```

For patch generation:

```python
BIOPSY_NAME = "biopsy_number"

NDPI_PATH = r"path/to/your/slide.ndpi"

MASK_PATH = r"path/to/your/mask.png"
```

The patch size, stride and output directories can also be
configured according to the dataset preparation requirements.

---

## Input

The data preparation modules require:

- Whole Slide Images
- QuPath GeoJSON annotations
- corresponding biopsy information

The Whole Slide Images are read using OpenSlide.

The annotations are provided as polygon geometries exported from
QuPath.

---

## Output

The preparation pipeline generates:

```text
Dataset_IA/
│
├── images/
│   ├── patch_001.png
│   ├── patch_002.png
│   └── ...
│
└── masks/
    ├── patch_001.png
    ├── patch_002.png
    └── ...
```

Each image patch has a corresponding segmentation-mask patch.

---

## Dataset classes

The segmentation masks use the tissue classes defined for the
project.

The classes used for model training are:

| Class | Description |
|---|---|
| Fond | Background / non-tissue regions |
| CK14 | CK14-positive tissue compartment |
| Stroma | Stromal tissue |
| Autres | Other tissue regions |

The class encoding must remain consistent between mask generation,
model training and inference.

---

## Quality control

The generated patches should be verified before model training.

The `verification_patches.py` module in the `evaluation/` directory
can be used to check the correspondence between image patches and
their associated segmentation masks.

This step helps identify potential errors in the prepared dataset
before training the segmentation model.

---

## Data confidentiality

The original Whole Slide Images, biopsy data, QuPath annotations
and other laboratory datasets are not included in the repository.

The paths used in the scripts are examples and must be replaced
with local paths to the authorized project data.

---

## Dependencies

The main libraries used by these modules include:

- Python
- OpenSlide
- NumPy
- OpenCV
- GeoPandas, when applicable
- Shapely, when applicable
- Pillow

See the main project `requirements.txt` for the complete list of
dependencies.

---

## Project context

These modules are part of the
**P16-KLHL35 AI Annotation Pipeline**.

Their purpose is to transform manually annotated histological
data into a structured image segmentation dataset suitable for
training deep learning models.

### Author : Hamza Graïn
