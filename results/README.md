# Results and Evaluation

This directory contains the results and evaluation procedures used to assess the P16-KLHL35 AI Annotation Pipeline.

The evaluation was designed to assess the model from several complementary perspectives:

- semantic segmentation performance,
- generalization to previously unseen biopsies,
- quantitative agreement between manual and AI annotations,
- spatial agreement between manual and AI annotations,
- intra-annotator reproducibility,
- practical time savings during annotation.

The objective was not to rely on a single segmentation metric, but to evaluate whether the generated annotations were sufficiently consistent with manual annotations for the intended quantitative analysis of tissue compartments.

---

## Evaluation Strategy

The evaluation was performed at several levels.

```text
Dataset verification
        ↓
Patch-level model evaluation
        ↓
Independent biopsy evaluation
        ↓
Manual vs AI quantitative concordance
        ↓
Manual vs AI spatial concordance
        ↓
Manual intra-annotator reproducibility
        ↓
Practical time-saving evaluation
````

Each evaluation addresses a different aspect of the pipeline.

## 1. Dataset and Patch Verification

The generated image patches and corresponding segmentation masks were checked before model training.

The verification procedure was used to ensure that:
```
image patches and masks correspond to the same spatial region,
patch dimensions are consistent,
the expected segmentation classes are present,
no obvious inconsistencies were introduced during dataset preparation.
````
The dataset was generated from manually annotated biopsies and progressively expanded as additional biopsies were annotated.

## Methodological limitation

The available annotated dataset was limited.

Only a subset of the available biopsies could be manually annotated because manual annotation was highly time-consuming. Consequently, the internal model validation was performed using a patch-level 80/20 split.

This approach allowed the available annotated data to be exploited for model development and monitoring, but it has an important limitation:

patches originating from the same biopsy may be present in both the training and validation subsets.

Therefore, the patch-level validation results should not be interpreted as an independent biopsy-level assessment of model generalization.

The 80/20 patch split was retained as a practical compromise given the limited amount of manually annotated data.

To address this limitation, additional evaluation was performed at the biopsy level using biopsies excluded from the corresponding training datasets.

## 2. Model Performance

Several model versions were trained using progressively larger sets of annotated biopsies.


| Model | Training biopsies | Excluded biopsies |
|---|---|---|
| V1 | 001, 002 | 003, 004, 005, 006, 007, 008 |
| V2 | 001, 002, 003, 006 | 004, 005, 007, 008 |
| V3 | 001, 002, 003, 006, 007, 008 | 004, 005 |
| V4 | 001, 002, 003, 004, 006, 007, 008 | 005 |

The model performance was evaluated using:
````
Dice coefficient,
Intersection over Union (IoU),
mean Dice,
mean IoU,
validation loss.
````
The evaluation was performed separately for the four tissue classes:
````
Fond
CK14
Stroma
Autres
````
The corresponding evaluation scripts are located in this directory.

## 3. Independent Biopsy Evaluation

To complement the patch-level validation, biopsies excluded from the corresponding training datasets were used for independent evaluation.

This evaluation was performed at the biopsy level rather than by randomly splitting patches.

The purpose was to assess how the model behaves when applied to an entire biopsy that was not used during training.

This distinction is important because the final application of the pipeline is performed on whole-slide images rather than isolated randomly selected patches.

The independent biopsy evaluation was not used as a training criterion or as a basis for selecting individual training samples.

## 4. Quantitative Concordance

Because the final objective of the project is to support quantitative analysis of tissue compartments, global tissue proportions were compared between manual and AI annotations.

For each biopsy, the proportions of:
````
CK14,
Stroma,
Autres
````
were calculated for both manual and AI-generated annotations.

The signed error was calculated as the difference between the AI and manual proportions.

The mean absolute error across the evaluated classes was then used to summarize the global quantitative difference.

A quantitative concordance value was calculated as:
````
Quantitative concordance = 100 - mean absolute error
````
This evaluation provides a complementary measure to pixel-level or polygon-level overlap metrics.

A model may have small local spatial differences while still estimating the global proportion of a tissue compartment accurately.

This distinction is particularly relevant for the intended downstream use of the annotations.

## 5. Spatial Concordance

Spatial agreement between manual and AI-generated annotations was evaluated using polygon geometries.

For each tissue class, the manual and AI polygons were compared using:
````
Intersection,
False-positive area,
False-negative area,
Precision,
Recall,
Dice coefficient,
Intersection over Union (IoU).
````
The polygons corresponding to each class were merged before comparison so that the evaluation focused on semantic tissue regions rather than on matching individual annotation objects.

This evaluation measures how well the AI reproduces the spatial localization of the tissue compartments.

## Complementarity with quantitative concordance

Quantitative and spatial concordance measure different properties.

For example, an AI annotation can reproduce the global proportion of a tissue compartment very closely while showing differences in the exact spatial localization of that compartment.

Both evaluations were therefore retained to distinguish:
````
how much tissue was assigned to a compartment, and
where that tissue was assigned.
````
## 6. Manual Intra-Annotator Reproducibility

A manual re-annotation of one biopsy was performed to estimate intra-annotator variability.

The second annotation was performed approximately two months after the initial annotation.

The same quantitative and spatial evaluation procedures were then applied between:
````
Manual annotation 1
        ↕
Manual annotation 2
````
This provides a reference for the variability associated with manual annotation itself.

The results showed that global compartment proportions were highly reproducible, while spatial agreement was lower, particularly for the Autres class.

Small clustered structures and artefacts could also be grouped differently during manual annotation, contributing to local spatial differences.

This evaluation should be interpreted as an estimate of intra-annotator variability for one re-annotated biopsy, rather than as a complete characterization of manual annotation variability across the entire dataset.

## 7. AI vs Manual Comparison

The independent biopsy evaluation and the intra-annotator evaluation were considered together to contextualize the AI results.

For example, on the independently evaluated biopsy 005:
````
manual vs AI quantitative concordance: 99.09%
manual 1 vs manual 2 quantitative concordance: 99.18%
````
The comparison suggests that the global compartment proportions generated by the model were close to the variability observed between two manual annotations for this particular evaluation.

However, spatial agreement remained lower for the AI annotation than for the manual re-annotation.

This distinction reinforces the importance of evaluating both quantitative and spatial agreement rather than relying on a single metric.

## 8. Time-Saving Evaluation

The practical benefit of the automated pipeline was evaluated by comparing the time required for manual annotation with the time required for:
````
AI prediction,
QuPath import,
manual correction of the generated annotations.
````
The total AI-assisted time was calculated as:
````
Total AI-assisted time =
AI inference time
+ QuPath import time
+ manual correction time
````
The relative time gain was then calculated by comparing the total AI-assisted time with the corresponding manual annotation time.

Representative biopsies included different levels of annotation complexity.

| Biopsy | Type	| Manual annotation | Total AI-assisted time | Time gain |
|---|---|---|---|---|
| 003 | Test | 18 h | 21 min 02 s | 98.1% |
| 004 | Test | 14 h | 20 min 14 s | 97.6% |
| 005 | Independent | 11 h | 29 min 53 s | 95.5% |
| 007 | Complex | 45 h | 84 min 57 s | 96.85% |

The evaluation highlights the potential reduction in annotation time provided by the AI-assisted workflow.

The correction step was intentionally included because the system is designed as an automated first annotation that can subsequently be reviewed and corrected rather than as a replacement for expert review.

## 9. Evaluation Summary

The different evaluations address complementary questions:

Evaluation	Main question
Patch verification	Is the prepared dataset internally consistent?
Patch-level validation	How does the model perform on the validation patches?
Independent biopsy evaluation	How does the model generalize to unseen biopsies?
Quantitative concordance	Does the AI reproduce global tissue proportions?
Spatial concordance	Does the AI reproduce tissue localization?
Intra-annotator reproducibility	How variable is manual annotation itself?
Time-saving evaluation	How much annotation time can the workflow reduce?

No single metric was considered sufficient to characterize the complete performance of the pipeline.

## 10. Limitations

Several methodological limitations should be considered when interpreting the results.

## Limited annotated dataset

Manual annotation required substantial time and limited the number of biopsies available for model development.

Only a subset of the complete dataset was manually annotated.

## Patch-level validation

The internal 80/20 validation was performed at the patch level.

Because patches originating from the same biopsy may occur in both subsets, this evaluation does not provide a fully independent estimate of biopsy-level generalization.

It was retained as a practical compromise to make use of the limited annotated data.

## Limited number of independent biopsies

The independent biopsy evaluation was performed on a limited number of biopsies.

The results therefore provide evidence of model behavior on previously unseen biopsies but should not be interpreted as a definitive characterization of generalization across a larger and more heterogeneous cohort.

## Intra-annotator evaluation

Manual reproducibility was evaluated through the re-annotation of one biopsy.

This provides useful information about annotation variability but cannot capture the full variability that could occur across multiple annotators and a larger number of biopsies.

## Spatial vs quantitative agreement

High quantitative concordance does not necessarily imply identical spatial localization.

The two evaluation approaches were therefore kept separate.

## AI-assisted workflow

The generated annotations are intended as a first annotation proposal.

Manual review and correction remain part of the workflow, particularly when tissue boundaries or small structures are difficult to segment.

## 11. Reproducibility

The scripts used for the different evaluations are included in this directory.

The evaluation results were generated from the project datasets and annotations.

The original whole-slide images, biological data, manual annotations and other sensitive research data are not included in this repository.

The repository therefore documents the evaluation methodology and software without distributing the underlying biological data.

## Project Context

This directory is part of the P16-KLHL35 AI Annotation Pipeline, developed during a research internship focused on the automated segmentation and annotation of tissue compartments in histological images.

The evaluation was designed to assess both the technical performance of the segmentation model and its practical usefulness for quantitative tissue analysis.
