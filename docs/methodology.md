# Methodology

## 1. Project Overview

The P16-KLHL35 AI Annotation Pipeline was developed to assist the annotation and quantitative analysis of tissue compartments in histological whole-slide images (WSIs).

The objective is to develop a deep learning-based semantic segmentation pipeline capable of generating an initial tissue compartment annotation that can subsequently be reviewed and corrected in QuPath.

The pipeline is designed around four semantic classes:

- `Fond`
- `CK14`
- `Stroma`
- `Autres`

The generated segmentation is converted into polygon annotations that can be imported into QuPath for visualization, correction, and quantitative analysis.

---

## 2. Dataset

The complete project dataset contains:
````
- 40 biopsies in total;
- 20 P16-stained biopsies;
- 20 KLHL35-stained biopsies;
- 20 paired serial-section samples.
````
A subset of the biopsies was manually annotated to create the training and evaluation dataset.

Manual annotation was performed using QuPath.

The manually annotated dataset contained:
````
- 8 annotated biopsies;
- 11,697 manual annotations;
- approximately 180 hours of manual annotation;
- 3,111 generated image patches.
````
The four semantic classes used for segmentation were:

```text
Fond
CK14
Stroma
Autres
````
The original histological images, annotations, and biological data are not distributed in this repository.

## 3. Annotation and Ground Truth Generation

Manual annotations were created in QuPath and used as the reference segmentation for model development.

The QuPath annotations were exported as GeoJSON files.

The GeoJSON annotations were converted into pixel-level segmentation masks using the data preparation pipeline.

The resulting masks assign a class label to each pixel according to the manually defined tissue compartment.

The conversion workflow is:
````
QuPath annotations
        ↓
GeoJSON export
        ↓
GeoJSON → segmentation mask
        ↓
Image / mask correspondence
        ↓
Patch generation
````
The conversion is implemented in:
````
src/data_preparation/geojson_to_mask.py
````
## 4. Patch Generation

Whole-slide images are too large to process directly with the segmentation model.

The WSIs are therefore divided into smaller image patches.

The preprocessing configuration used for the project includes:
````
WSI reading at level 4;
patch size: 512 × 512 pixels;
stride: 256 pixels;
overlapping patches.
````
Each image patch is associated with its corresponding ground-truth segmentation mask.

The resulting dataset is organized into paired image and mask patches.
````
Whole Slide Image
        +
Ground-truth mask
        ↓
Patch extraction
        ↓
Image patches + mask patches
````
The patch generation procedure is implemented in:
````
src/data_preparation/generate_patches.py
````
A dedicated verification script is used to check the correspondence between generated images and masks.

## 5. Data Augmentation and Preprocessing

The training pipeline applies image preprocessing and data augmentation to improve the robustness of the model.

The exact transformations are defined in the training scripts.

The image data are converted into the format expected by the neural network before being passed to the model.

The segmentation masks retain their discrete class labels throughout preprocessing.

Data augmentation is applied consistently to the image and corresponding segmentation mask so that their spatial correspondence is preserved.

## 6. Model Architecture

The segmentation model is based on a U-Net architecture with a ResNet34 encoder.
````
Input image
     ↓
ResNet34 encoder
     ↓
Feature extraction
     ↓
U-Net decoder
     ↓
Pixel-level classification
     ↓
4 semantic classes
````
U-Net provides the encoder-decoder structure required for pixel-level segmentation, while the ResNet34 encoder extracts hierarchical image features.

The model produces a class prediction for each pixel of an input patch.

## 7. Training

The model was trained using a combination of Cross-Entropy Loss and Dice Loss.

The training objective was defined as a weighted combination of the two losses:
````
Total Loss = 0.5 × Cross-Entropy Loss
           + 0.5 × Dice Loss
````
Class weights were introduced to account for differences in class representation:
````
Fond    : 0.5
CK14    : 1.0
Stroma  : 1.0
Autres  : 1.2
````
The optimizer and other training parameters are defined in the training scripts.

Training was performed for a maximum of 300 epochs with early stopping based on validation performance.

The early stopping patience was set to 10 epochs for the final experiments.

## 8. Validation Strategy

Two complementary evaluation levels were used.

### 8.1 Patch-level validation

An 80/20 split was used at the patch level for internal training validation.

This validation was useful for:
````
monitoring the training process;
comparing training and validation loss;
detecting major overfitting behaviour;
calculating Dice and IoU metrics during training.
````
However, this split has an important methodological limitation.

Because the split was performed at the patch level, patches originating from the same biopsy may be present in both the training and validation subsets.

Consequently, this validation does not constitute a fully independent biopsy-level evaluation.

This strategy was retained as a practical compromise because the number of manually annotated biopsies was limited and manual annotation was highly time-consuming.

### 8.2 Independent biopsy evaluation

To complement the patch-level validation, several model versions were evaluated on biopsies that were excluded from their corresponding training datasets.

The training configurations were progressively expanded:

| Model | Training biopsies | Excluded biopsies |
|---|---|---|
| V1 | 001, 002 | 003, 004, 005, 006, 007, 008 |
| V2 | 001, 002, 003, 006 | 004, 005, 007, 008 |
| V3 | 001, 002, 003, 006, 007, 008 | 004, 005 |
| V4 | 001, 002, 003, 004, 006, 007, 008 | 005 |

The independent biopsy evaluation was designed to provide a more representative assessment of model behaviour on previously unseen biopsies.

## 9. Segmentation Metrics

Model performance was evaluated using class-level segmentation metrics.

### Dice coefficient

The Dice coefficient measures the overlap between the predicted segmentation and the reference segmentation.

It was calculated independently for each tissue class.

### Intersection over Union

IoU was also calculated for each class.

Both metrics were used because they provide complementary descriptions of segmentation overlap.

The mean Dice and mean IoU across the four classes were also reported as global performance indicators.

## 10. Whole-Slide Inference

After training, the model can be applied to a new whole-slide image.

The WSI is processed using overlapping patches with the same patch dimensions used during training.

Each patch is independently passed through the segmentation model.

The resulting predictions are then reconstructed at slide level.

The inference workflow is:
````
Whole Slide Image
        ↓
Patch extraction
        ↓
Model prediction
        ↓
Overlapping prediction aggregation
        ↓
Slide-level segmentation mask
````
The inference procedure is implemented in:
````
src/inference/predict_wsi.py
````

## 11. Post-processing

The reconstructed segmentation is processed before polygon extraction.

The post-processing stage includes:
````
confidence filtering;
morphological processing;
removal of very small regions.
````
The purpose is to reduce isolated prediction artefacts and obtain more coherent tissue regions.

The processed segmentation mask is then used for polygon extraction.

## 12. Polygon Extraction and QuPath Integration

The final segmentation mask is converted into polygon geometries.

Contours are extracted independently for each semantic class.

The resulting polygons are processed and exported as GeoJSON annotations.

The general workflow is:
```
Segmentation mask
        ↓
Contour extraction
        ↓
Polygon generation
        ↓
Polygon processing
        ↓
GeoJSON export
        ↓
QuPath import
````
The complete implementation is located in:
````
src/inference/predict_geojson_to_QuPath/
````
The generated GeoJSON can be imported into QuPath to visualize the predicted tissue compartments and perform subsequent manual corrections.

The AI output is therefore considered a first annotation proposal rather than a final diagnostic or annotation decision.

## 13. Quantitative Concordance

Because the intended application includes quantitative tissue analysis, model evaluation was not limited to pixel-level segmentation metrics.

The global proportion of each tissue compartment was calculated for both manual and AI annotations.

The evaluated classes were:
````
CK14;
Stroma;
Autres.
````
The difference between AI and manual proportions was calculated for each class.

The mean absolute error across the evaluated classes was then used to summarize the global quantitative difference.

The quantitative concordance was defined as:
````
Quantitative concordance = 100 - mean absolute error
````
This evaluation measures whether the AI preserves the overall distribution of tissue compartments, independently of small local spatial differences.

## 14. Spatial Concordance

A separate spatial evaluation was performed by comparing the manual and AI-generated polygon geometries.

For each tissue class, the following metrics were calculated:
````
Precision;
Recall;
Dice;
IoU.
````
The evaluation was based on the continuous area of the predicted and reference polygons.

This analysis provides information about the spatial localization of the predicted tissue compartments.

Spatial concordance and quantitative concordance were treated as complementary measurements.

A prediction may reproduce the global proportion of a compartment accurately while still differing locally in its exact spatial boundaries.

## 15. Manual Intra-Annotator Reproducibility

To estimate the variability associated with manual annotation, one biopsy was manually annotated a second time.

The second annotation was performed approximately two months after the first annotation.

The same quantitative and spatial comparison procedures were applied between the two manual annotations.

This provides an estimate of intra-annotator reproducibility and provides context for interpreting the differences observed between manual and AI annotations.

The analysis was limited to one re-annotated biopsy and therefore does not represent the complete variability that could exist between multiple annotators or across the entire dataset.

## 16. Time-Saving Evaluation

The practical benefit of the automated pipeline was evaluated by comparing manual annotation time with AI-assisted annotation time.

The AI-assisted workflow included:
````
AI inference;
QuPath import;
manual correction of the generated annotations.
````
The total AI-assisted time was therefore calculated as:
````
AI-assisted time =
Inference time
+ QuPath import time
+ Manual correction time
````
The time gain was calculated by comparing this total with the time required for the corresponding manual annotation.

This evaluation was performed on biopsies with different levels of annotation complexity.

The purpose was to measure the practical reduction in annotation workload rather than to assign a monetary value to the time saved.

## 17. Evaluation Philosophy

The evaluation strategy was designed to avoid relying on a single performance indicator.

The different analyses answer different questions:

Evaluation	Question addressed
Patch-level validation	How does the model behave on the internal validation patches?
Independent biopsy evaluation	How does the model behave on unseen biopsies?
Dice / IoU	How well do predicted and reference regions overlap?
Quantitative concordance	Are global tissue proportions preserved?
Spatial concordance	Are tissue compartments located in similar regions?
Intra-annotator reproducibility	How variable is manual annotation itself?
Time-saving evaluation	How much manual annotation time can be reduced?

This multi-level evaluation is particularly important given the limited number of manually annotated biopsies.

## 18. Methodological Limitations

The main limitations of the methodology are:

### Limited annotated dataset

Only a subset of the complete biopsy dataset was manually annotated because manual annotation required substantial time.

### Patch-level validation bias

The 80/20 validation split was performed at the patch level rather than strictly at the biopsy level.

Consequently, patches from the same biopsy may occur in both training and validation subsets.

The resulting validation metrics may therefore provide an optimistic estimate of generalization compared with a strictly biopsy-independent validation.

The approach was retained as a practical compromise to exploit the limited manually annotated dataset.

### Limited independent evaluation

The number of independent biopsies available for evaluation remained limited.

The independent biopsy results should therefore be interpreted as an evaluation of behaviour on the available unseen biopsies rather than as a definitive estimate of performance on a larger population.

### Manual annotation variability

Manual annotation itself contains variability, particularly for small or ambiguous regions.

The intra-annotator evaluation was performed on one biopsy and therefore provides only a limited estimate of this variability.

### Post-processing effects

The final polygon geometry may differ slightly from the raw pixel-level segmentation because of post-processing and polygon conversion.

Consequently, visual inspection of the segmentation mask and evaluation of the final QuPath polygons address related but not identical representations of the model output.

## 19. Reproducibility and Data Protection

The methodology and evaluation scripts are included in this repository to document the computational workflow.

The original:
````
whole-slide images;
histological images;
biological data;
manual annotations;
GeoJSON annotations;
segmentation masks;
````
are not distributed in the repository.

The trained model weights are also not included.

This repository therefore focuses on the software architecture, methodology, and evaluation procedures while keeping the underlying research data confidential.
