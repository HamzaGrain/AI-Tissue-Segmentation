# AI-Tissue-Segmentation

# Automated Histological Annotation Pipeline

## P16-KLHL35

Pipeline d'annotation automatique de compartiments tissulaires
sur Whole Slide Images (WSI) à l'aide d'un modèle de segmentation
sémantique pour un but de quantification de biomarqueurs P16/KLHL35 
dans le suite du projet.

### Objectifs

- Automatiser l'annotation de compartiments tissulaires
- Réduire le temps nécessaire à l'annotation manuelle
- Générer des annotations compatibles avec QuPath
- Évaluer la concordance entre annotations manuelles et IA

## Pipeline

Image WSI ->
Découpage en patches ->
Segmentation par IA ->
Post-processing ->
Extraction des polygones ->
Traitement géométrique ->
Export GeoJSON ->
QuPath ->
Correction / validation
