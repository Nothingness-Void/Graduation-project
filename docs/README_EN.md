<p align="center">
  <a href="../README.md">简体中文</a> ·
  <a href="README_EN.md">English</a> ·
  <a href="README_JA.md">日本語</a>
</p>

# QSAR Prediction Model for Huggins Parameter (χ) Based on Molecular Descriptors

> ⚠️ This version was translated by AI and may contain errors.

> This project uses **QSAR (Quantitative Structure-Activity Relationship)** methods to predict the **Huggins parameter (χ)** of polymer-solvent systems using molecular descriptors and ML/DNN models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Full Pipeline](#full-pipeline)
  - [Step 1: Obtain SMILES](#step-1-obtain-smiles-molecular-representations)
  - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
  - [Step 2.5: Dataset Merging](#step-25-dataset-merging)
  - [Step 3: Feature Engineering](#step-3-feature-engineering)
  - [Step 4: Feature Selection](#step-4-two-stage-feature-selection)
  - [Step 5: Model Training & AutoTuning](#step-5-model-training--auto-tuning)
  - [Step 6: Model Validation & Analysis](#step-6-model-validation--analysis)
- [Data Files](#data-files)
- [Model Performance Benchmarks](#model-performance-benchmarks)
- [Dependencies](#dependencies)

---

## Overview

The **Huggins parameter (χ)** is a key thermodynamic parameter describing polymer-solvent interactions, reflecting the affinity between solvent and polymer in a mixed system.

Core workflow:

1. Extract compound names from literature data and convert to **SMILES** molecular representations
2. Merge multi-source datasets (old: 323 + new: 1586 = **1,893 samples**)
3. Compute all **~210** 2D molecular descriptors using **RDKit**, plus fingerprint similarities and interaction features → **320-dimensional feature matrix**
4. Use **Genetic Algorithm (GA)** to select optimal feature subset from 320 dimensions
5. Train ML/DNN models with **AutoTune** hyperparameter optimization using selected features

---

## Project Structure

```
Graduation-project/
│
├── 获取SMILES.py              # Step 1: Compound name → SMILES
├── 数据处理部分代码.py          # Step 2: χ expression parsing + temperature expansion
├── 合并数据集.py               # Step 2.5: Merge old and new datasets
├── 特征工程.py                 # Step 3: Full RDKit descriptor extraction (320-dim)
├── 特征筛选.py                 # Step 4a: RFECV feature selection
├── 遗传.py                    # Step 4b: Genetic Algorithm (GA) feature selection
├── feature_config.py           # Feature config center (unified feature column management)
│
├── DNN.py                     # Step 5a: DNN deep neural network modeling
├── DNN_AutoTune.py            # Step 5b: DNN Hyperband auto-tuning
├── Sklearn.py                 # Step 5c: Sklearn Bayesian optimization modeling
├── Sklearn_AutoTune.py        # Step 5d: Sklearn RandomizedSearch auto-tuning
│
├── DNN_模型验证.py             # Step 6a: DNN model validation
├── DNN特征贡献分析.py          # Step 6c: DNN SHAP feature contribution analysis
├── Y_Randomization.py         # Step 6d: Y-Randomization (Y-Scrambling) test
│
├── Huggins.xlsx               # Raw data: compound names + Huggins parameters
│
├── data/                      # Intermediate data
│   ├── smiles_raw.csv
│   ├── smiles_cleaned.xlsx
│   ├── huggins_preprocessed.xlsx
│   ├── 43579_2022_237_MOESM1_ESM.csv  # External dataset (1,586 entries)
│   ├── merged_dataset.csv             # Merged dataset (1,893 entries)
│   ├── molecular_features.xlsx        # 320-dim feature matrix
│   └── features_optimized.xlsx        # Selected feature subset
│
├── results/                   # Models & results
│   ├── dnn_model.keras
│   ├── dnn_preprocess.pkl
│   ├── sklearn_model_bundle.pkl
│   ├── ga_best_model.pkl
│   ├── ga_selected_features.txt
│   ├── ga_evolution_log.csv
│   ├── sklearn_tuning_summary.csv
│   ├── train_test_split_indices.npz   # Unified train/test split indices
│   ├── feature_selection.png
│   └── dnn_loss.png
│
├── final_results/             # Final deliverables (separated from intermediates)
│   └── sklearn/
│       ├── sklearn_model_bundle.pkl
│       ├── fingerprint_model.pkl
│       ├── sklearn_tuning_summary.csv
│       ├── sklearn_validation_results.xlsx
│       ├── sklearn_feature_importance.csv
│       ├── sklearn_feature_importance.png
│       ├── sklearn_validation_plots.png
│       ├── y_randomization.png
│       ├── y_randomization.csv
│       └── sklearn_final_report.txt
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
conda install -c conda-forge rdkit  # RDKit must be installed via conda
```

### Key Dependencies

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data processing & scientific computing |
| `rdkit` | Molecular descriptor computation, fingerprint generation |
| `scikit-learn` | Traditional ML models & data preprocessing |
| `scikit-optimize` | Bayesian hyperparameter optimization (BayesSearchCV) |
| `xgboost` | XGBoost regression model |
| `deap` | Genetic Algorithm feature selection |
| `tensorflow` / `keras` | Deep Neural Network (DNN) |
| `keras-tuner` | DNN Hyperband auto-tuning |
| `shap` | Model interpretability analysis (SHAP values) |
| `joblib` | Model serialization |
| `matplotlib` | Data visualization |
| `requests` / `tqdm` | HTTP requests / progress bars |

---

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Complete Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Huggins.xlsx ─────────────────┐                                    │
│       │                        │                                    │
│       ▼                        │                                    │
│  Step 1: 获取SMILES.py         │                                    │
│       │                        │                                    │
│       ▼                        │                                    │
│  Step 2: 数据处理部分代码.py   │                                    │
│       │                        │                                    │
│       ▼                        ▼                                    │
│  Step 2.5: 合并数据集.py ◄─── New data (ESM.csv)                    │
│       │                                                             │
│       ▼                                                             │
│  Step 3: 特征工程.py → 320-dim full RDKit descriptors               │
│       │                                                             │
│       ▼                                                             │
│  Step 4a: 遗传.py (GA coarse: 320 → ~20-40)                        │
│       │                                                             │
│       ▼                                                             │
│  Step 4b: 特征筛选.py (RFECV fine: ~20-40 → ~8-15)                  │
│       │                                                             │
│       ├─────────────────────┐                                       │
│       ▼                     ▼                                       │
│  Step 5a: Sklearn       Step 5b: DNN                                │
│  (Sklearn_AutoTune.py)  (DNN.py / DNN_AutoTune.py)                  │
│       │                     │                                       │
│       ▼                     ▼                                       │
│  Step 6: Validation + Feature Contribution Analysis                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Obtain SMILES Molecular Representations

**Script**: [`获取SMILES.py`](获取SMILES.py)

**Function**: Converts compound names from `Huggins.xlsx` to SMILES molecular structure strings via PubChem / NCI API.

```bash
python 获取SMILES.py
```

> ⚠️ Requires internet access to query PubChem and NCI databases.

---

### Step 2: Data Preprocessing

**Script**: [`数据处理部分代码.py`](数据处理部分代码.py)

**Function**: Handles temperature-dependent χ expressions (e.g., `0.43+34.7T`), temperature expansion (20-50°C), and outlier filtering (`-1 < χ < 5`).

```bash
python 数据处理部分代码.py
```

---

### Step 2.5: Dataset Merging

**Script**: [`合并数据集.py`](合并数据集.py)

**Function**: Merges old data (`huggins_preprocessed.xlsx`, 323 entries) with new external data (`43579_2022_237_MOESM1_ESM.csv`, 1,586 entries) into a unified format. After deduplication: **1,893 samples**.

**Data flow**: Old data + New data → `data/merged_dataset.csv`

**Unified columns**: `Polymer, Solvent, Polymer_SMILES, Solvent_SMILES, chi, temperature, source`

```bash
python 合并数据集.py
```

---

### Step 3: Feature Engineering

**Script**: [`特征工程.py`](特征工程.py)

**Function**: Uses RDKit's `CalcMolDescriptors()` to extract all **~210 2D molecular descriptors** for both polymer and solvent, then adds fingerprint similarity and interaction features.

**Data flow**: `data/merged_dataset.csv` → `data/molecular_features.xlsx`

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| Polymer descriptors (suffix `_1`) | ~148 | MolWt, LogP, TPSA, fragment counts, topological indices, etc. |
| Solvent descriptors (suffix `_2`) | ~155 | Same as above |
| Fingerprint similarity | 3 | Avalon, Morgan, Topological |
| Interaction features | 14 | Delta_LogP, Delta_TPSA, HB_Match, Inv_T, etc. |
| **Total** | **~320** | After cleaning (removing high-missing + constant columns) |

**Special handling**: `[*]` connection point markers in polymer SMILES are replaced with `[H]` for proper RDKit parsing.

```bash
python 特征工程.py
```

---

### Step 4: Two-Stage Feature Selection

Uses a **GA coarse screening → RFECV fine screening** two-stage strategy to progressively select optimal features from 320 dimensions:

```
320-dim ──GA coarse──→ ~20-40 dim ──RFECV fine──→ ~8-15 dim ──→ Modeling
```

#### Step 4a: Genetic Algorithm (GA) Coarse Screening

**Script**: [`遗传.py`](遗传.py)

**Function**: Uses DEAP genetic algorithm to globally search for optimal feature subsets from ~320 dimensions. GA can explore nonlinear feature combination effects, suitable for high-dimensional coarse screening.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population size | 100 | 100 candidates per generation |
| Max generations | 60 | Upper limit (usually early-stopped) |
| Early stopping | 12 generations no improvement | Automatic stop |
| CV folds | 3 | Balance speed and accuracy |
| Estimator | RF(n=100, depth=8) | Lightweight and fast |
| Feature count constraint | [5, 40] | Control model complexity |

**Output**: `results/ga_selected_features.txt`, `results/ga_evolution_log.csv`, `results/train_test_split_indices.npz`, auto-updates `feature_config.py`

> ℹ️ GA creates and saves train/test split indices. All downstream scripts automatically reuse the same split, ensuring complete test set isolation.

```bash
python 遗传.py    # ~20-40 minutes
```

#### Step 4b: RFECV Fine Screening

**Script**: [`特征筛选.py`](特征筛选.py)

**Function**: From GA-selected ~20-40 features, uses RFECV to iteratively remove redundant features and pinpoint the optimal subset. Automatically reads GA results from `feature_config.py`.

> ⚠️ Must run `遗传.py` first. Automatically loads GA-saved train/test split indices and performs selection only on the training set.

**Output**: Auto-updates `feature_config.py` and `data/features_optimized.xlsx`

```bash
python 特征筛选.py
```

#### Unified Feature Management

**Script**: [`feature_config.py`](feature_config.py)

Feature selection results are stored in this file, defining `SELECTED_FEATURE_COLS` (selected features) for use by downstream training and validation scripts.

---

### Step 5: Model Training & Auto-Tuning

#### Step 5a: DNN Deep Neural Network

**Script**: [`DNN.py`](DNN.py)

| Config | Value |
|--------|-------|
| Architecture | 48 → BN → Dropout(0.15) → 24 → BN → Dropout(0.1) → 12(L2) → 1 |
| Loss function | Huber |
| Training strategy | Train with 5 random seeds, select best |
| Data split | 60% train / 20% validation / 20% test |
| Normalization | StandardScaler on both X and y |

```bash
.venv\Scripts\python.exe DNN.py
```

#### Step 5b: DNN Hyperband Auto-Tuning

**Script**: [`DNN_AutoTune.py`](DNN_AutoTune.py)

Uses Keras Tuner's Hyperband algorithm to search for optimal DNN architecture (1-3 layers, 12-64 units, learning rate, regularization, etc.).

```bash
.venv\Scripts\python.exe DNN_AutoTune.py
```

#### Step 5c: Sklearn Traditional Machine Learning

**Script**: [`Sklearn.py`](Sklearn.py)

Batch trains multiple Sklearn regression models using BayesSearchCV for optimal parameter search.

#### Step 5d: Sklearn AutoTune (Recommended)

**Script**: [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py)

4 models × 50 parameter sets × 5-fold CV automatic optimization:

| Model | Search Dimensions |
|-------|-------------------|
| GradientBoosting | loss, lr, n_estimators, depth, subsample |
| XGBRegressor | lr, n_estimators, depth, reg_alpha/lambda |
| RandomForest | n_estimators, depth, max_features |
| MLPRegressor | hidden layers, activation, alpha, lr |

After execution, automatically completes:

1. Best model search (CV model selection)
2. Test set validation (R²/MAE/RMSE, using only unseen test data)
3. Feature contribution analysis (built-in importance or permutation importance)
4. Validation visualization (Actual vs Predicted, residual distribution, model comparison — 4 plots)
5. Final deliverables output to `final_results/sklearn/`

```bash
python Sklearn_AutoTune.py
```

---

### Step 6: Model Validation & Analysis

#### Model Validation

| Script | Function |
|--------|----------|
| [`DNN_模型验证.py`](DNN_模型验证.py) | Load DNN model and evaluate R²/MAE/RMSE on full data |
| [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py) | Automatically outputs Sklearn validation results after training (`final_results/sklearn/sklearn_validation_results.xlsx`) |

#### Feature Contribution Analysis

| Script | Function |
|--------|----------|
| [`DNN特征贡献分析.py`](DNN特征贡献分析.py) | SHAP GradientExplainer for DNN feature contributions |
| [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py) | Automatically outputs Sklearn feature importance after training (`final_results/sklearn/sklearn_feature_importance.*`) |

#### Y-Randomization Test

**Script**: [`Y_Randomization.py`](Y_Randomization.py)

**Function**: Y-Scrambling validation — shuffles y values 100 times and retrains the model to verify whether the QSAR model truly learned feature-target relationships. If real model R² is significantly higher than randomized distribution (p < 0.05), the model is valid.

**Output**: `final_results/sklearn/y_randomization.png`, `y_randomization.csv`

```bash
python Y_Randomization.py
```

---

## Data Files

| File | Location | Description | Stage |
|------|----------|-------------|-------|
| `Huggins.xlsx` | Root | Raw data | Input |
| `43579_2022_237_MOESM1_ESM.csv` | `data/` | External dataset (1,586 entries) | Input |
| `smiles_raw.csv` | `data/` | SMILES query results | Step 1 |
| `smiles_cleaned.xlsx` | `data/` | Manually cleaned SMILES | Manual |
| `huggins_preprocessed.xlsx` | `data/` | Preprocessed data (323 entries) | Step 2 |
| `merged_dataset.csv` | `data/` | Merged dataset (1,893 entries) | Step 2.5 |
| `molecular_features.xlsx` | `data/` | 320-dim feature matrix | Step 3 |
| `features_optimized.xlsx` | `data/` | Selected feature subset | Step 4 |
| `ga_selected_features.txt` | `results/` | GA-selected feature list | Step 4a |
| `ga_evolution_log.csv` | `results/` | GA evolution log | Step 4a |
| `sklearn_model_bundle.pkl` | `results/` | Sklearn unified model bundle | Step 5 |
| `dnn_model.keras` | `results/` | DNN model | Step 5 |
| `train_test_split_indices.npz` | `results/` | Unified train/test split indices | Step 4a |
| `sklearn_final_report.txt` | `final_results/sklearn/` | Sklearn final report | Step 5d |
| `sklearn_validation_results.xlsx` | `final_results/sklearn/` | Sklearn validation details | Step 5d |
| `sklearn_feature_importance.png` | `final_results/sklearn/` | Sklearn feature importance plot | Step 5d |
| `sklearn_validation_plots.png` | `final_results/sklearn/` | Sklearn validation plots (4 subplots) | Step 5d |
| `y_randomization.png` | `final_results/sklearn/` | Y-Randomization R² distribution | Step 6 |
| `y_randomization.csv` | `final_results/sklearn/` | Y-Randomization detailed data | Step 6 |

---

## Model Performance Benchmarks

> Results from AutoTune on merged dataset (1,886 samples, 6 features via RFECV)

| Model | CV Val R² | Test R² | Test MAE | Test RMSE |
|-------|-----------|---------|----------|-----------|
| **GradientBoosting** | **0.749** | **0.812** | 0.156 | 0.263 |
| XGBRegressor | 0.726 | 0.799 | 0.150 | 0.271 |
| RandomForest | 0.692 | 0.780 | 0.177 | 0.284 |
| MLPRegressor | 0.616 | 0.725 | 0.208 | 0.318 |
| DNN (Keras) | — | 0.649 | 0.240 | 0.359 |

> ℹ️ All models are evaluated on the same test set. The test set does not participate in feature selection or model training.
> 💡 Performance is expected to improve further after GA selects the optimal feature subset from 320 dimensions.

---

## Quick Start

```bash
# 1. Clone the project
git clone <repository-url>
cd Graduation-project

# 2. Install dependencies
pip install -r requirements.txt
conda install -c conda-forge rdkit

# 3. Dataset merging + Feature engineering + Two-stage feature selection + Modeling
python 合并数据集.py              # Merge old and new data
python 特征工程.py                # Full RDKit descriptors (320-dim)
python 遗传.py                   # GA coarse screening (320 → ~20-40, ~20-40 min)
python 特征筛选.py                # RFECV fine screening (~20-40 → ~8-15)
python Sklearn_AutoTune.py       # Sklearn auto-tuning

# Or: if you already have data/molecular_features.xlsx, start from Step 4
python 遗传.py
python Sklearn_AutoTune.py
```

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **R²** | 1 - SS_res/SS_tot | Coefficient of determination (closer to 1 = better) |
| **MAE** | mean(\|y_true - y_pred\|) | Mean Absolute Error |
| **RMSE** | √(mean((y_true - y_pred)²)) | Root Mean Squared Error |

---

## License

This project is a graduation thesis project, for academic research purposes only.
