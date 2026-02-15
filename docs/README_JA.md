<p align="center">
  <a href="../README.md">简体中文</a> ·
  <a href="README_EN.md">English</a> ·
  <a href="README_JA.md">日本語</a>
</p>

# 分子記述子に基づく Huggins パラメータ（chi）の QSAR 予測

> この版は AI 翻訳です。表現に軽微な不自然さが含まれる場合があります。

## 目次

- [概要](#概要)
- [プロジェクト構成](#プロジェクト構成)
- [パイプライン](#パイプライン)
- [Step 5: 自動チューニング](#step-5-自動チューニング)
- [Step 6: 検証と解析](#step-6-検証と解析)
- [主要出力](#主要出力)
- [クイックスタート](#クイックスタート)
- [評価指標](#評価指標)

## 概要

本プロジェクトは、高分子-溶媒系の Huggins パラメータ（chi）を QSAR 手法で予測します。

1. SMILES データの整備
2. RDKit 記述子と相互作用特徴量の作成
3. 二段階特徴量選択（GA 粗選別 + RFECV 精選別）
4. Sklearn/DNN の自動チューニング
5. テストセット検証と Y-randomization

## プロジェクト構成

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

## パイプライン

1. `获取SMILES.py`: 化合物名から SMILES へ変換
2. `数据处理部分代码.py`: chi-温度式の前処理
3. `合并数据集.py`: 旧データと外部データを統合
4. `特征工程.py`: 記述子行列を作成
5. `遗传.py`: GA 粗選別 + split 保存
6. `特征筛选.py`: RFECV 精選別（訓練側のみ）
7. `Sklearn_AutoTune.py` / `DNN_AutoTune.py`: モデル自動探索
8. `Y_Randomization.py` / `DNN_Y_Randomization.py`: 妥当性検証

## Step 5: 自動チューニング

- `Sklearn_AutoTune.py`: 複数回帰器を CV で比較し最適化
- `DNN_AutoTune.py`: Hyperband による構造探索 + 多シード再学習
- 両者とも `results/train_test_split_indices.npz` を共通再利用

## Step 6: 検証と解析

- `Y_Randomization.py`: sklearn 最良モデルの Y-scrambling
- `DNN_Y_Randomization.py`: DNN の Y-scrambling
- `DNN_模型验证.py`: 保存済み DNN の評価
- `DNN特征贡献分析.py`: DNN の SHAP 解析

## 主要出力

- split: `results/train_test_split_indices.npz`
- 特徴量設定: `feature_config.py`
- sklearn 最終レポート: `final_results/sklearn/sklearn_final_report.txt`
- sklearn Y-randomization: `final_results/sklearn/y_randomization.csv`
- DNN サマリ: `results/tuner_summary.txt`
- DNN Y-randomization: `final_results/dnn/dnn_y_randomization_summary.txt`

## クイックスタート

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

## 評価指標

- `R2`: 決定係数
- `MAE`: 平均絶対誤差
- `RMSE`: 二乗平均平方根誤差
