# XAIguiFormer — Explainable AI Guided Transformer for Brain Disorder Identification

This repository contains the implementation of **XAIguiFormer**, based on the ICLR 2025 paper [*XAIguiFormer: Explainable Artificial Intelligence Guided Transformer for Brain Disorder Identification*](https://openreview.net/forum?id=AD5yx2xq8R), extended with support for the **BEED (Brainwave EEG Dataset)** — a tabular 4-class brainwave classification dataset.

![XAIguiFormer Architecture](XAIguiFormer.png)

---

## Table of Contents

- [Abstract](#abstract)
- [Architecture Overview](#architecture-overview)
- [BEED Dataset](#beed-dataset)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
  - [BEED Dataset Preparation](#beed-dataset-preparation)
  - [Other Datasets (TDBRAIN / TUAB / ds004504)](#other-datasets-tdbrain--tuab--ds004504)
- [Training](#training)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [License](#license)

---

## Abstract

EEG-based connectomes offer a low-cost and portable method to identify brain disorders using deep learning. With growing interest in model interpretability, Explainable Artificial Intelligence (XAI) is widely applied to understand deep learning decisions. However, most research focuses solely on interpretability analysis, overlooking XAI's potential to **improve model performance**.

XAIguiFormer bridges this gap with a dynamical-system-inspired architecture where XAI not only provides explanations but also enhances the transformer by **refining self-attention** to capture more relevant dependency relationships. Key innovations include:

- **Connectome Tokenizer** — treats single-band graphs as atomic tokens to generate a frequency-domain sequence without destroying the connectome's topological structure.
- **2D Rotary Frequency Encoding (dRoFE)** — integrates frequency band information and subject demographics (age, gender) into tokens via a rotation matrix.
- **XAI-Guided Attention** — uses DeepLift attributions to dynamically guide query/key projections in the self-attention mechanism.

---

## Architecture Overview

The model pipeline consists of three core components:

### 1. Connectome Encoder (`modules/connectome_encoder.py`)
Processes raw graph data (EEG channels as nodes, functional connectivity as edges) using stacked **GINEConv** GNN layers. Produces a per-frequency-band graph embedding via mean pooling.

### 2. XAIgui Transformer Encoder (`modules/transformer.py`)
Operates on the sequence of frequency-band embeddings produced by the Connectome Encoder:
- Custom multi-head self-attention (**XAIguiAttention**) with optional XAI-guided query/key refinement.
- **dRoFE** positional encoding that encodes both frequency band boundaries and demographic information.
- Classification head with average pooling + MLP.

### 3. Explainer Module (`modules/explainer.py`)
Uses **DeepLift** (via Captum) to compute frequency-band attributions from an initial forward pass. These attributions are fed back into the transformer for a second, enhanced forward pass — producing a more informed prediction.

**Forward Pass Flow:**
```
Raw Graph → Connectome Encoder → Initial Transformer Pass → DeepLift Explainer
                                                                    ↓
                                              Explanation-Guided Transformer Pass → Final Prediction
```

---

## BEED Dataset

### Overview

The **BEED (Brainwave EEG Dataset)** is a tabular dataset (`BEED_Data.csv`) used for **4-class brainwave state classification**. Unlike the other datasets in this benchmark (TDBRAIN, TUAB, ds004504) which consist of raw multi-channel EEG recordings, BEED provides pre-extracted features in a flat CSV format.

### Dataset Characteristics

| Property | Value |
|---|---|
| **Format** | CSV (tabular) |
| **Features** | 16 numerical columns (`X1` – `X16`) |
| **Target** | `y` — integer class label (0, 1, 2, 3) |
| **Number of Classes** | 4 |
| **Total Samples** | ~8,002 rows |
| **Task** | Multi-class brainwave state classification |

### Adaptation to XAIguiFormer

Since XAIguiFormer expects graph-structured EEG connectomes, the BEED tabular data is converted to **pseudo-connectomes** via the `utils/prepare_BEED.py` script:

1. **Node Features (Coherence substitute):** Each row's 16 features become the diagonal of a 16×16 coherence matrix (1 frequency band).
2. **Edge Features (wPLI substitute):** A fully connected graph is constructed using a **Gaussian RBF kernel** on the normalized feature pairs: `edge_weight = exp(-γ · (xi - xj)²)`.
3. **Demographics:** Placeholder values `[age=50, gender=0]` (since BEED has no demographic metadata).
4. **Labels:** One-hot encoded 4-class labels.

### Data Split

The data is split **block-wise per class** (not randomly) to prevent time-series leakage:

| Split | Proportion |
|---|---|
| Train | 60% |
| Validation | 20% |
| Test | 20% |

Features are **Z-score normalized** using statistics computed solely on the training set.

---

## Repository Structure

```
XAIguiFormer/
├── main.py                          # Training & evaluation entry point
├── config.py                        # Default configuration (YACS CfgNode)
├── logger.py                        # TensorBoard writer & logger setup
├── environment.yml                  # Conda environment specification
│
├── configs/                         # Dataset-specific YAML configurations
│   ├── BEED_model.yaml              # BEED: 4-class, 16 nodes, 1 freq band
│   ├── TDBRAIN_model.yaml           # TDBRAIN config
│   ├── TUAB_model.yaml              # TUAB config
│   ├── ds004504_model.yaml          # ds004504 config
│   ├── TDBRAIN_preprocess.yaml      # TDBRAIN preprocessing config
│   └── TUAB_preprocess.yaml         # TUAB preprocessing config
│
├── models/
│   └── XAIguiFormer.py              # Main model class
│
├── modules/                         # Model building blocks
│   ├── connectome_encoder.py        # GNN-based graph encoder
│   ├── transformer.py               # XAIgui Transformer with custom attention
│   ├── positional_encoding_wrapper.py  # dRoFE: 2D rotary freq encoding + demographics
│   ├── explainer.py                 # Captum-based DeepLift explainer
│   ├── explainer_wrapper.py         # Wrapper for explainer integration
│   ├── gnn.py                       # GNN layer definitions
│   ├── gnn_wrapper.py               # GNN layer wrapper utilities
│   ├── mlp.py                       # MLP / feed-forward layers
│   └── activation.py               # Custom activations (GeGLU)
│
├── data/
│   ├── EEGBenchmarkDataset.py       # PyG InMemoryDataset for all datasets
│   └── transform.py                 # Data augmentation (Mixup / CutMix)
│
├── utils/
│   ├── prepare_BEED.py              # Convert BEED CSV → pseudo-connectome .npy files
│   ├── prepare_ds004504.py          # Prepare ds004504 dataset
│   ├── preprocessing.py             # Raw EEG preprocessing pipeline
│   ├── constructFC.py               # Functional connectivity construction
│   ├── transform_dataformAndlabel.py  # Data format & label transformations
│   ├── eval_metrics.py              # BAC, Sensitivity, AUCPR, AUROC metrics
│   └── visualizer.py               # Activation cache for XAI visualization
│
├── EEGBenchmarkDataset/             # Root data directory (generated)
│   └── BEED/
│       ├── raw/
│       │   ├── train/               # Training subjects
│       │   ├── val/                 # Validation subjects
│       │   └── test/                # Test subjects
│       └── processed/               # Cached PyG datasets (auto-generated)
│
├── output/                          # Training logs, TensorBoard, saved models
│   └── results/
│       └── BEED/
│
├── XAIguiFormer.png                 # Architecture diagram
└── model_documentation.md           # Detailed model documentation
```

---

## Environment Setup

### Option A: From the environment file

```bash
conda env create -f environment.yml
conda activate XAIguiFormer
```

### Option B: Step-by-step installation

```bash
conda create --name XAIguiFormer python=3.10
conda activate XAIguiFormer

# Core deep learning
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pytorch-scatter -c pyg

# Transformer & training utilities
conda install -c conda-forge -c pytorch -c defaults timm
conda install -c conda-forge torchmetrics
conda install captum -c pytorch

# Supporting libraries
conda install tensorboard
conda install -c conda-forge einops
conda install -c conda-forge yacs
conda install pandas matplotlib
conda install bytecode
```

> **Note:** For EEG preprocessing (raw data → connectomes), you will additionally need `mne`, `mne-connectivity`, and `mne-icalabel`. See [Data Preparation](#other-datasets-tdbrain--tuab--ds004504).

---

## Data Preparation

### BEED Dataset Preparation

1. **Obtain the BEED dataset** — place `BEED_Data.csv` in a known location.

2. **Update the paths** in `utils/prepare_BEED.py`:
   ```python
   csv_file = "/path/to/BEED_Data.csv"
   output_directory = "/path/to/XAIguiFormer/EEGBenchmarkDataset/BEED/raw"
   ```

3. **Run the preparation script:**
   ```bash
   python utils/prepare_BEED.py
   ```

   This generates the following structure under `EEGBenchmarkDataset/BEED/raw/`:
   ```
   train/
   ├── row_00000/
   │   ├── row_00000_EC_coherence.npy    # (1, 16, 16, 1)
   │   ├── row_00000_EC_wpli.npy         # (1, 16, 16, 1)
   │   ├── row_00000_EC_demographics.npy # (2, 1)
   │   └── row_00000_EC_label.npy        # (1, 4) one-hot
   ├── row_00001/
   │   └── ...
   val/
   └── ...
   test/
   └── ...
   ```

4. **First training run** will auto-generate `processed/*.pt` files via PyTorch Geometric's `InMemoryDataset`.

### Other Datasets (TDBRAIN / TUAB / ds004504)

For raw EEG datasets, create a separate MNE environment and run the preprocessing pipeline:

```bash
conda create -c conda-forge --strict-channel-priority --name=mne mne
conda activate mne
conda install -c conda-forge mne-connectivity mne-icalabel
```

1. Update paths in `configs/*_preprocess.yaml`
2. Run preprocessing: `python utils/preprocessing.py`
3. Construct connectomes: `python utils/constructFC.py`
4. Organize files: `python utils/transform_dataformAndlabel.py`

---

## Training

Train the model by specifying the dataset name:

```bash
# Train on BEED (4-class brainwave classification)
python main.py --dataset BEED

# Train on other supported datasets
python main.py --dataset TDBRAIN
python main.py --dataset TUAB
python main.py --dataset ds004504
```

Training outputs are saved to `output/results/<dataset>/`.

### Key BEED Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 128 |
| Learning Rate | 1e-4 (AdamW) |
| Weight Decay | 0.01 |
| GNN Layers | 2 |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| Hidden Dimension | 64 |
| Dropout | 0.1 |
| Explainer | DeepLift |
| Label Smoothing | 0.1 |
| Mixup / CutMix | Disabled |
| Loss Alpha (auxiliary) | 0.5 |

To modify hyperparameters, edit `configs/BEED_model.yaml`.

---

## Evaluation Metrics

The model is evaluated using the following metrics (computed via `torchmetrics`):

| Metric | Description |
|---|---|
| **Balanced Accuracy (BAC)** | Average of sensitivity and specificity |
| **Sensitivity (Recall)** | True positive rate across classes |
| **AUCPR** | Area Under the Precision-Recall Curve |
| **AUROC** | Area Under the ROC Curve |

Subject-level ensemble is performed during validation/test: predictions from all segments of the same subject are averaged before computing metrics.

The model also outputs **XAI frequency band contribution scores** — the average absolute DeepLift attribution per frequency band — providing interpretable insights into which spectral components drive predictions.

---

## Citation

If you find this work useful, please consider citing the original paper:

```bibtex
@inproceedings{
  guo2025XAIguiFormer,
  title={XAIguiFormer: explainable artificial intelligence guided transformer for brain disorder identification},
  author={Hanning Guo, Farah Abdellatif, Yu Fu, N. Jon Shah, Abigail Morrison, J\"{u}rgen Dammers},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=AD5yx2xq8R}
}
```

---

## License

Please refer to the original [XAIguiFormer repository](https://openreview.net/forum?id=AD5yx2xq8R) for licensing terms.
