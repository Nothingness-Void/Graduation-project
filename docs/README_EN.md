<p align="center">
  <a href="../README.md">简体中文</a> ·
  <a href="README_EN.md">English</a> ·
  <a href="README_JA.md">日本語</a>
</p>

# QSAR Prediction of Huggins Parameter (chi) from Molecular Descriptors

> This version is AI-translated and may contain minor wording errors.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Step 5: Auto-Tuning](#step-5-auto-tuning)
- [Step 6: Validation and Analysis](#step-6-validation-and-analysis)
- [Key Outputs](#key-outputs)
- [Quick Start](#quick-start)
- [Metrics](#metrics)

## Overview

This project predicts the Huggins parameter (chi) for polymer-solvent systems using QSAR workflow:

1. Build/clean SMILES data.
2. Engineer RDKit descriptors and interaction features.
3. Run two-stage feature selection: GA coarse search then RFECV refinement.
4. Train AutoTune models (Sklearn and DNN).
5. Validate with held-out test set and Y-randomization.

## Project Structure

```text
Graduation-project/
├── 获取SMILES.py
├── 数据处理部分代码.py
├── 合并数据集.py
├── 特征工程.py
├── 遗传.py
├── 特征筛选.py
├── feature_config.py
├── Sklearn_AutoTune.py
├── DNN_AutoTune.py
├── Y_Randomization.py
├── DNN_Y_Randomization.py
├── DNN_模型验证.py
├── DNN特征贡献分析.py
├── utils/data_utils.py
├── data/
├── results/
└── final_results/
```

## Pipeline

1. `获取SMILES.py`: compound name to SMILES.
2. `数据处理部分代码.py`: preprocess chi-temperature expressions.
3. `合并数据集.py`: merge legacy and external datasets.
4. `特征工程.py`: generate full descriptor matrix.
5. `遗传.py`: GA feature coarse search and save split indices.
6. `特征筛选.py`: RFECV refinement on training split only.
7. `Sklearn_AutoTune.py` / `DNN_AutoTune.py`: model auto-tuning.
8. `Y_Randomization.py` / `DNN_Y_Randomization.py`: statistical validity checks.

## Step 5: Auto-Tuning

- `Sklearn_AutoTune.py`: compares multiple regressors with randomized search and CV.
- `DNN_AutoTune.py`: Hyperband-based architecture search, then multi-seed retraining.
- Both reuse the same saved train/test split (`results/train_test_split_indices.npz`).

## Step 6: Validation and Analysis

- `Y_Randomization.py`: Y-scrambling for the sklearn best model.
- `DNN_Y_Randomization.py`: Y-scrambling for DNN using the same split.
- `DNN_模型验证.py`: metric evaluation for saved DNN model.
- `DNN特征贡献分析.py`: SHAP-based DNN feature contribution analysis.

## Key Outputs

- Split indices: `results/train_test_split_indices.npz`
- Feature list: `feature_config.py`
- Sklearn final report: `final_results/sklearn/sklearn_final_report.txt`
- Sklearn Y-randomization: `final_results/sklearn/y_randomization.csv`
- DNN summary: `results/tuner_summary.txt`
- DNN Y-randomization: `final_results/dnn/dnn_y_randomization_summary.txt`

## Quick Start

```bash
pip install -r requirements.txt

python 合并数据集.py
python 特征工程.py
python 遗传.py
python 特征筛选.py
python Sklearn_AutoTune.py
python DNN_AutoTune.py
python Y_Randomization.py
python DNN_Y_Randomization.py
```

## Metrics

- `R2`: goodness of fit.
- `MAE`: mean absolute error.
- `RMSE`: root mean squared error.
