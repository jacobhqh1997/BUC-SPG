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

## Pre-requisites and Environment

### Our Environment
* Linux (Tested on Ubuntu 24.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX A6000)
* Python (3.12.6), PyTorch (version 2.0.0), Lifelines (version 0.27.8), NumPy (version 1.24.1),MONAI (version 1.3), Pandas (version 2.1.2), Albumentations (version 1.3.1), OpenCV (version 4.8.1), Pillow (version 9.3.0), OpenSlide (version 1.1.2), Captum (version 0.6.0), SciPy (version 1.11.3), Seaborn (version 0.13.0), Matplotlib (version 3.8.1), torch_geometric (version 2.4.0), torch-scatter (version 2.1.2), torch-sparse (version 0.6.18).
### Environment Configuration
1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
   ```bash
   $ conda create -n env python=3.12.6
   $ conda activate env
   $ pip install torch
   ```
      *Note:  `pip install` command is required for Pytorch installation.*
   
2. To try out the Python code and set up environment, please activate the `env` environment first:

   ``` shell
   $ conda activate env
   ```
3. For ease of use, you can just set up the environment and run the following:
   ``` shell
   $ pip install -r requirements.txt
   ```
---

### 2. TINSPGNet (Multimodal Fusion Model)
- **Model's Intended Use**: To integrate features from multiple modalities ( pathology, text) and predict patient prognosis (e.g., survival risk).
- **Inputs**:
    - Pre-extracted feature vectors from the upstream models:
        - Macroscopic features (`.pt`)
        - Microscopic features (`.pt`)
        - Textual features (`.pt`)
- **Outputs**:
    - A risk score for patient prognosis.
- **Potential Failure Modes**:
    - **Feature Quality**: The model's performance is influenced by the quality of the input features. Inaccuracies from upstream modules can affect the final prediction.
- **Compute Footprint & Latency**:
    - **Training**: Trained on a single NVIDIA GeForce RTX A6000 (48GB) GPU.
    - **Inference**: Inference requires few seconds as it operates on pre-computed feature vectors.

## Data Format

WSIs and clinical information of patients are used in this project. Raw WSIs are stored as ```.svs```, ```.mrxs``` or ```.tiff``` files. Clinical information are stored as ```.csv``` files. CT images are stored in NIFIT format (nii.gz)

## Data Preparation


### Generate structured pathology report

structured pathology reports are derived from structured ground-truth data (e.g., TNM stage, grade, LVI status).

```shell
$ cd ./Data_process
$ python UC_report_standardization_ground_generation.py
```
### Generate ** knowledge-guided patch representation**

- Create original macroscopic tissue probability heatmaps. WSIs are first processed by UCSparseNet network to get  local  probability heatmaps, Global knowledge-guided patch representation: create global probability heatmaps. 

``` shell
  $ cd ./Data_prepare
  $ python  UCSparseNet_inference.py
  $ python  inference_global_probability.py 
```

* To cut the empty area of combined  tissue probability heatmaps and get square input for MacroContextNet training

    ``` shell
    $ cd ./Data_process
    $ python cut_heatmap.py
    ```

## Feature_extractor

- Subsequently, we generated macroscopic feature, textual feature, radiographic feature, and microscopic features for TINSPGNet training, respectively. 

  ```bash
  $ cd ./Feature_extractor
  $ python micro_feature.py   #get Uni microscopic feature
  $ python macro_feature.py   #get  macroscopic feature
  $ python text_feature.py   #get textual feature
  ```

### Training Scripts



In Training Scripts, the train_macro_cash.py script is used to train the macroscopic module.

```shell
$ cd ./Training Scripts
$ python train_macro_cash.py   # MacroContextNet training scripts 
```

In Training Scripts, the train_TINSPGNet.py script is used to train the TINSPGNet multimodal model.

```bash
$ cd ./Training Scripts
$ python train_TINSPGNet.py  # TINSPGNet training scripts 
```

```bash
$ cd ./Training Scripts
$ python geo_refine_process.py  #get refined segmentation results
```

## Biomarker_quantification

- Run the code in Biomarker_quantification to generate the corresponding marker calculation score

  ```bash
  $ cd ./Biomarker_quantification
  $ python Coloc_M.py #get Coloc_M score
  $ python IMTS.py #get IMTS score
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
         ├── macro_file                                 + + +
                ├── train                                   +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                ├── valid                                   +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                └── ...                                     +    
         └── feature_file                                   +
                ├── Micro                               + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +
                ├── macro                               + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +    
                ├── text                                + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +                          
```
DATA_ROOT is the base directory of all datasets (e.g. the directory to your SSD or HDD). DATASET is the name of the folder containing data specific to one experiment.


## Acknowledgements
- Prognosis training and test code base structure was inspired by [[PathFinder]](https://github.com/Biooptics2021/PathFinder) and[[MCAT]](https://github.com/mahmoodlab/MCAT) .

  



