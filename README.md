
# ***Spatial multi-omics prior-guided multimodal representation learning for biomarker discovery and progression-risk prediction in bladder cancer***

© This code is made available for non-commercial academic purposes.

## Overview

Bladder Urothelial carcinoma (BUC), remains a prevalent and lethal malignancy within the urinary tract. Precise  BUC progression risk stratification is critical for accurate personalized therapy. Here we present a spatial multi-omics prior-guided multimodal representation learning for biomarker discovery and progression prediction in bladder cancer

## Directory Structure

* **Training Scripts**: *Training Scripts for  TINSPGNet.*
* **Data_process**: *Data preprocessing file.*
* **Feature_extractor**: * textual, and microscopic feature extraction.*
* **Biomarker_core**: Detailed code definitions for each Biomarker
* **model_architectures**: Detailed model architectures
* **inference**: Contextual CTP-Net and TNSL-Net inference and probability heatmaps.
* **bioinformatics**: Spatial preprocessing, ecosystem discovery, differentiation scoring and patient-level comparisons.

## Software and Environment

The project uses separate environments for multimodal pathology, spatial
bioinformatics and R-based survival analysis. See [software_versions.txt](software_versions.txt)
for recorded analysis versions and the dated inventory of the available environments.
The inventory is documentation, not a single installation lock file.

The public code explains the methods and model interfaces. Private patient data,
pathology annotations and trained checkpoints are not included; study-specific
inputs and compatible software environments are required for execution.

---

### 2. TINSPGNet (Multimodal Fusion Model)

- **Model's Intended Use**: To integrate features from multiple modalities ( niche-prior, tissue-prior, micro feature, text) and predict patient prognosis (e.g., progression survival risk).
- **Inputs**:
  - Pre-extracted feature vectors from the upstream models:
    - niche-prior (`.npy`)
    - tissue-prior (`.npy`)    
    - Microscopic features (`.pt`)
    - Textual features (`.pt`)
- **Outputs**:
  - A risk score for patient progression-risk prediction.
- **Potential Failure Modes**:
  - **Feature Quality**: The model's performance is influenced by the quality of the input features. Inaccuracies from upstream modules can affect the final prediction.
- **Compute Footprint & Latency**:
  - **Training**: Trained on a single NVIDIA GeForce RTX A6000 (48GB) GPU.
  - **Inference**: Inference requires few seconds as it operates on pre-computed feature vectors.

## Data Format

WSIs and clinical information of patients are used in this project. Raw WSIs are stored as ``.svs``, ``.mrxs`` or ``.tiff`` files. Clinical information are stored as ``.csv`` files.

## Data Preparation

### Generate structured pathology report

structured pathology reports are derived from structured ground-truth data (e.g., TNM stage, grade, LVI status).

```shell
$ cd ./Data_process
$ python UC_report_standardization_ground_generation.py
```

### Generate spatial probability heatmaps

Frozen UNI embeddings and registered patch-grid coordinates are assembled into
centre-plus-eight-neighbour inputs. CTP-Net produces an **H × W × 8 tissue
probability map**. TNSL-Net predicts **H × W × 4 N1–N4 niche probabilities** at
tumour centres, retaining the surrounding tissue as context. Continuous probabilities
are preserved; unevaluated positions are masked. See [inference/README.md](inference/README.md)
for the input contract and function-level example.

## Spatial Bioinformatics

The biological workflow connects high-definition spatial profiling to the priors
used by the histology models:

1. **Preprocessing and annotation:** segmented-cell QC, count preservation,
   normalization and log transformation; scVI batch integration with a 10-dimensional
   latent space; marker-reviewed labels and confidence-filtered logistic propagation.
2. **Spatial domains and ecosystems:** specimen-specific Delaunay graphs and
   three-layer CellCharter aggregation define whole-tissue domains. Tumour anchors
   within reviewed regions are described using 112-µm neighbourhoods, latent
   features and cell composition, followed by scaling, PCA and full-covariance
   mixture modeling. Reviewed solutions comprise 12 domains and four ecosystems.
3. **Histology supervision and differentiation:** anchor posterior probabilities
   are averaged within registered tiles containing at least five anchors. Three
   epithelial program scores define a relative differentiation coordinate; pooled
   probability-weighted ecosystem coordinates provide a fixed reference for
   projecting TNSL-Net predictions.
4. **Spatial comparisons:** specimen-level core–front comparisons use exact
   sign-flip tests. ADC-target expression is compared between muscle-proximal
   and distal compartments using paired Wilcoxon tests and BH correction, with
   proximal thresholds of 24, 32 (primary), 40 and 48 µm. Public cohorts provide
   complementary patient-level spatial comparisons.

The code is organized into `preprocessing.py`, `ecosystems.py`,
`differentiation.py`, `spatial_comparisons.py` and `workflow.py`.
These are methodological illustrations; pathology review,
registration and private reference inputs remain external. Cell annotations,
three differentiation programs and four ecosystem labels are distinct objects.


## Feature_extractor

- Subsequently, we generated  textual feature, and microscopic features for TINSPGNet training, respectively.

  ```bash
  $ cd ./Feature_extractor
  $ python micro_feature.py   #get Uni microscopic feature
  $ python text_feature.py   #get textual feature
  ```

### Training Scripts

In Training Scripts, the train_CTP_Net.py script is used to train the tissue-prior module.

```shell
$ cd ./Training Scripts
$ python train_CTP_Net.py   #  training scripts 
```
In Training Scripts, the train_TNSL_Net.py script is used to train the tumor-ecosystem-prior module.

```shell
$ cd ./Training Scripts
$ python train_TNSL_Net.py   #  training scripts 
```

In Training Scripts, the train_TINSPGNet.py script is used to train the TINSPGNet multimodal model.

```bash
$ cd ./Training Scripts
$ python train_TINSPGNet.py  # TINSPGNet training scripts 
```


## Biomarker_quantification

- Run the code in Biomarker_quantification to generate the corresponding marker calculation score

  ```bash
  $ cd ./Biomarker_quantification
  $ python Coloc-M.py #get Coloc-M score
  $ python IMTS.py #get IMTS score
  $ python MIRI.py #get MIRI score
  $ python TMIST.py #get TMIST score 
  ```

### Data Distribution

```bash
DATA_ROOT/
    └──DATASET/
         ├── clinical_information                       + + + 
                ├── train.csv                               +
                ├── valid.csv                               +
                └── ...                                     +
         ├── WSI_data                                       +
                ├── train                                   +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                Source WSI file
                       └── ...                              +
                ├──valid                                    +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                      +
                       └── ...                              +
                └── ...                                 + + +
         └── feature_file                                   +
                ├── tissue-prior                            + + +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                ├── niche-prior                             + + +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +  
                ├── text                                + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +    
                ├── micro                                + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +                      
```

DATA_ROOT is the base directory of all datasets (e.g. the directory to your SSD or HDD). DATASET is the name of the folder containing data specific to one experiment.

## Acknowledgements

- Prognosis training and test code base structure was inspired by [[PathFinder]](https://github.com/Biooptics2021/PathFinder) and[[MCAT]](https://github.com/mahmoodlab/MCAT) .
