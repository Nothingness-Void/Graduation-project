<p align="center">
  <a href="../README.md">简体中文</a> ·

  <a href="README_EN.md">English</a> ·
  <a href="README_JA.md">日本語</a>
</p>

# 分子記述子に基づくハギンスパラメータ（χ）の QSAR 予測モデル

> ⚠️ このバージョンはAIによって翻訳されたものであり、誤りが含まれている可能性があります。

> 本プロジェクトは **QSAR（定量的構造活性相関）** 手法を用い、分子記述子と機械学習 / 深層学習モデルにより、高分子-溶媒系の **ハギンスパラメータ（χ）** を予測します。

---

## 📋 目次

- [概要](#概要)
- [プロジェクト構成](#プロジェクト構成)
- [環境構築](#環境構築)
- [パイプライン全体図](#パイプライン全体図)
  - [Step 1：SMILES 取得](#step-1smiles-分子表現の取得)
  - [Step 2：データ前処理](#step-2データ前処理)
  - [Step 2.5：データ統合](#step-25データ統合)
  - [Step 3：特徴量エンジニアリング](#step-3特徴量エンジニアリング)
  - [Step 4：特徴量選択](#step-4二段階特徴量選択)
  - [Step 5：モデル学習と自動チューニング](#step-5モデル学習と自動チューニング)
  - [Step 6：モデル検証と分析](#step-6モデル検証と分析)
- [データファイル一覧](#データファイル一覧)
- [モデル性能ベンチマーク](#モデル性能ベンチマーク)
- [依存ライブラリ](#依存ライブラリ)

---

## 概要

**ハギンスパラメータ（χ）** は、高分子-溶媒相互作用を記述する重要な熱力学パラメータで、混合系における溶媒と高分子の親和性を反映します。

本プロジェクトのワークフロー：

1. 文献データから化合物名を抽出し、**SMILES** 分子構造表現に変換
2. 複数データソースを統合（旧データ 323 件 + 新データ 1,586 件 = **1,893 件**）
3. **RDKit** で全 **~210 個** の 2D 分子記述子を自動計算 + フィンガープリント類似度 + 相互作用特徴量 → **320 次元特徴量行列**
4. **遺伝的アルゴリズム（GA）** により 320 次元から最適な特徴量サブセットを選択
5. 選択された特徴量を基に **AutoTune** で ML / DNN モデルを自動ハイパーパラメータ最適化

---

## プロジェクト構成

```
Graduation-project/
│
├── 获取SMILES.py              # Step 1: 化合物名 → SMILES
├── 数据处理部分代码.py          # Step 2: χ 式の解析 + 温度展開
├── 合并数据集.py               # Step 2.5: 新旧データの統合
├── 特征工程.py                 # Step 3: 全量 RDKit 記述子抽出 (320 次元)
├── 特征筛选.py                 # Step 4a: RFECV 特徴量選択
├── 遗传.py                    # Step 4b: 遺伝的アルゴリズム (GA) 特徴量選択
├── feature_config.py           # 特徴量設定センター
│
├── DNN.py                     # Step 5a: DNN 深層ニューラルネットワーク
├── DNN_AutoTune.py            # Step 5b: DNN Hyperband 自動チューニング
├── Sklearn.py                 # Step 5c: Sklearn ベイズ最適化
├── Sklearn_AutoTune.py        # Step 5d: Sklearn ランダムサーチ自動チューニング
│
├── DNN_模型验证.py             # Step 6a: DNN モデル検証
├── DNN特征贡献分析.py          # Step 6c: DNN SHAP 特徴量寄与分析
├── Y_Randomization.py         # Step 6d: Y-Randomization テスト
│
├── Huggins.xlsx               # 元データ
├── data/                      # 中間データ
├── results/                   # モデル＆結果
├── final_results/             # 最終成果物
├── requirements.txt
└── README.md
```

---

## 環境構築

### 前提条件

- Python 3.8+
- pip パッケージマネージャ

### 依存関係のインストール

```bash
pip install -r requirements.txt
conda install -c conda-forge rdkit  # RDKit は conda 経由でインストール
```

### 主要ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `pandas` / `numpy` | データ処理・科学計算 |
| `rdkit` | 分子記述子計算、フィンガープリント生成 |
| `scikit-learn` | 従来型 ML モデル・前処理 |
| `scikit-optimize` | ベイズハイパーパラメータ最適化 |
| `xgboost` | XGBoost 回帰モデル |
| `deap` | 遺伝的アルゴリズム特徴量選択 |
| `tensorflow` / `keras` | 深層ニューラルネットワーク (DNN) |
| `keras-tuner` | DNN Hyperband 自動チューニング |
| `shap` | モデル解釈性分析（SHAP 値） |
| `matplotlib` | データ可視化 |

---

## パイプライン全体図

```
┌─────────────────────────────────────────────────────────────────────┐
│                       パイプライン全体図                              │
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
│  Step 2.5: 合并数据集.py ◄─── 新データ (ESM.csv)                    │
│       │                                                             │
│       ▼                                                             │
│  Step 3: 特征工程.py → 320 次元 RDKit 記述子                        │
│       │                                                             │
│       ▼                                                             │
│  Step 4a: 遗传.py (GA 粗篩: 320 → ~20-40)                          │
│       │                                                             │
│       ▼                                                             │
│  Step 4b: 特征筛选.py (RFECV 精篩: ~20-40 → ~8-15)                 │
│       │                                                             │
│       ├─────────────────────┐                                       │
│       ▼                     ▼                                       │
│  Step 5a: Sklearn       Step 5b: DNN                                │
│  (Sklearn_AutoTune.py)  (DNN.py / DNN_AutoTune.py)                  │
│       │                     │                                       │
│       ▼                     ▼                                       │
│  Step 6: モデル検証 + 特徴量寄与分析                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Step 1：SMILES 分子表現の取得

**スクリプト**: [`获取SMILES.py`](获取SMILES.py)

**機能**: `Huggins.xlsx` の化合物名を PubChem / NCI API 経由で SMILES 分子構造文字列に変換。

```bash
python 获取SMILES.py
```

> ⚠️ PubChem・NCI データベースへのネットワーク接続が必要

---

### Step 2：データ前処理

**スクリプト**: [`数据处理部分代码.py`](数据处理部分代码.py)

**機能**: χ 式の温度依存項処理（例：`0.43+34.7T`）、温度展開（20-50°C）、外れ値フィルタリング（`-1 < χ < 5`）。

```bash
python 数据处理部分代码.py
```

---

### Step 2.5：データ統合

**スクリプト**: [`合并数据集.py`](合并数据集.py)

**機能**: 旧データ（323 件）と新外部データ（1,586 件）を統一フォーマットに統合。重複除去後 **1,893 件**。

**データフロー**: 旧データ + 新データ → `data/merged_dataset.csv`

```bash
python 合并数据集.py
```

---

### Step 3：特徴量エンジニアリング

**スクリプト**: [`特征工程.py`](特征工程.py)

**機能**: RDKit の `CalcMolDescriptors()` で全 **~210 個の 2D 分子記述子** を高分子・溶媒それぞれで計算し、フィンガープリント類似度と相互作用特徴量を追加。

**データフロー**: `data/merged_dataset.csv` → `data/molecular_features.xlsx`

| 特徴量カテゴリ | 数量 | 説明 |
|--------------|------|------|
| 高分子記述子（接尾辞 `_1`） | ~148 | MolWt, LogP, TPSA 等 |
| 溶媒記述子（接尾辞 `_2`） | ~155 | 同上 |
| フィンガープリント類似度 | 3 | Avalon, Morgan, Topological |
| 相互作用特徴量 | 14 | Delta_LogP, Delta_TPSA, HB_Match, Inv_T 等 |
| **合計** | **~320** | クリーニング後 |

```bash
python 特征工程.py
```

---

### Step 4：二段階特徴量選択

**GA 粗篩 → RFECV 精篩** の二段階戦略：

```
320 次元 ──GA粗篩──→ ~20-40 次元 ──RFECV精篩──→ ~8-15 次元 ──→ モデリング
```

#### Step 4a：遺伝的アルゴリズム (GA) 粗篩

**スクリプト**: [`遗传.py`](遗传.py)

**機能**: DEAP 遺伝的アルゴリズムで ~320 次元から最適特徴量サブセットをグローバルに探索。

| パラメータ | 値 | 説明 |
|-----------|---|------|
| 母集団サイズ | 100 | 各世代 100 候補 |
| 最大世代数 | 60 | 上限（通常早期停止） |
| 早期停止 | 12 世代改善なし | 自動停止 |
| CV 分割数 | 3 | 速度と精度のバランス |
| 推定器 | RF(n=100, depth=8) | 軽量高速 |
| 特徴量数制約 | [5, 40] | モデル複雑度制御 |

**出力**: `results/ga_selected_features.txt`、`results/ga_evolution_log.csv`、`results/train_test_split_indices.npz`、`feature_config.py` を自動更新

> ℹ️ GA が train/test 分割インデックスを作成・保存。全ての下流スクリプトが同一分割を自動的に再利用し、テストセットの完全な隔離を保証します。

```bash
python 遗传.py    # 約 20-40 分
```

#### Step 4b：RFECV 精篩

**スクリプト**: [`特征筛选.py`](特征筛选.py)

**機能**: GA が選出した ~20-40 特徴量から RFECV で冗長特徴量を逐次除去し、最適サブセットを特定。

> ⚠️ 先に `遗传.py` を実行する必要があります。GA が保存した train/test 分割インデックスを自動読み込みし、訓練セットのみで選択を実行します。

```bash
python 特征筛选.py
```

#### 統一特徴量管理

**スクリプト**: [`feature_config.py`](feature_config.py)

特徴量選択結果は `SELECTED_FEATURE_COLS` として本ファイルに保存され、下流の学習・検証スクリプトが参照します。

---

### Step 5：モデル学習と自動チューニング

#### Step 5a：DNN 深層ニューラルネットワーク

**スクリプト**: [`DNN.py`](DNN.py)

| 設定項目 | 値 |
|---------|---|
| ネットワーク構造 | 48 → BN → Dropout(0.15) → 24 → BN → Dropout(0.1) → 12(L2) → 1 |
| 損失関数 | Huber |
| 学習戦略 | 5 種のランダムシードで複数回学習、最良を選択 |
| データ分割 | 60% 訓練 / 20% 検証 / 20% テスト |
| 標準化 | X, y 共に StandardScaler |

```bash
.venv\Scripts\python.exe DNN.py
```

#### Step 5b：DNN Hyperband 自動チューニング

**スクリプト**: [`DNN_AutoTune.py`](DNN_AutoTune.py)

Keras Tuner の Hyperband アルゴリズムで DNN 最適アーキテクチャを探索（1-3 層、12-64 ユニット、学習率、正則化等）。

```bash
.venv\Scripts\python.exe DNN_AutoTune.py
```

#### Step 5c：Sklearn 従来型機械学習

**スクリプト**: [`Sklearn.py`](Sklearn.py)

複数の Sklearn 回帰モデルを BayesSearchCV で一括学習。

#### Step 5d：Sklearn AutoTune（推奨）

**スクリプト**: [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py)

4 モデル × 50 パラメータセット × 5 分割 CV 自動最適化：

| モデル | 探索次元 |
|--------|---------|
| GradientBoosting | loss, lr, n_estimators, depth, subsample |
| XGBRegressor | lr, n_estimators, depth, reg_alpha/lambda |
| RandomForest | n_estimators, depth, max_features |
| MLPRegressor | hidden layers, activation, alpha, lr |

実行後の自動処理：

1. 最良モデル探索（CV モデル選択）
2. テストセット検証（R²/MAE/RMSE、訓練に未使用のテストデータのみ使用）
3. 特徴量寄与分析（組み込み重要度または permutation importance）
4. 検証可視化（Actual vs Predicted、残差分布、モデル比較 — 4 図）
5. 最終成果物を `final_results/sklearn/` に出力

```bash
python Sklearn_AutoTune.py
```

---

### Step 6：モデル検証と分析

#### モデル検証

| スクリプト | 機能 |
|-----------|------|
| [`DNN_模型验证.py`](DNN_模型验证.py) | DNN モデルの R²/MAE/RMSE 評価 |
| [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py) | 学習後に Sklearn 検証結果を自動出力 |

#### 特徴量寄与分析

| スクリプト | 機能 |
|-----------|------|
| [`DNN特征贡献分析.py`](DNN特征贡献分析.py) | SHAP GradientExplainer による DNN 特徴量寄与分析 |
| [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py) | 学習後に Sklearn 特徴量重要度を自動出力 |

#### Y-Randomization テスト

**スクリプト**: [`Y_Randomization.py`](Y_Randomization.py)

**機能**: Y-Scrambling 検証 — y 値を 100 回ランダムに打ち変えてモデルを再学習し、QSAR モデルが特徴量と目標値の真の関係を学習しているかを検証。実モデルの R² がランダム化分布より有意に高ければ（p < 0.05）、モデルは有効です。

**出力**: `final_results/sklearn/y_randomization.png`、`y_randomization.csv`

```bash
python Y_Randomization.py
```

---

## データファイル一覧

| ファイル | 場所 | 説明 | ステージ |
|---------|------|------|---------|
| `Huggins.xlsx` | ルート | 元データ | 入力 |
| `43579_2022_237_MOESM1_ESM.csv` | `data/` | 外部データセット (1,586 件) | 入力 |
| `molecular_features.xlsx` | `data/` | 320 次元特徴量行列 | Step 3 |
| `features_optimized.xlsx` | `data/` | 選択済み特徴量サブセット | Step 4 |
| `ga_selected_features.txt` | `results/` | GA 選択特徴量リスト | Step 4a |
| `train_test_split_indices.npz` | `results/` | 統一 train/test 分割インデックス | Step 4a |
| `sklearn_model_bundle.pkl` | `results/` | Sklearn 統一モデルバンドル | Step 5 |
| `dnn_model.keras` | `results/` | DNN モデル | Step 5 |
| `sklearn_validation_plots.png` | `final_results/sklearn/` | Sklearn 検証可視化 | Step 5d |
| `y_randomization.png` | `final_results/sklearn/` | Y-Randomization R² 分布図 | Step 6 |
| `y_randomization.csv` | `final_results/sklearn/` | Y-Randomization 詳細データ | Step 6 |

---

## モデル性能ベンチマーク

> 統合データセット（1,886 サンプル、RFECV 6 特徴量）での AutoTune 結果

| モデル | CV Val R² | Test R² | Test MAE | Test RMSE |
|--------|-----------|---------|----------|-----------|
| **GradientBoosting** | **0.749** | **0.812** | 0.156 | 0.263 |
| XGBRegressor | 0.726 | 0.799 | 0.150 | 0.271 |
| RandomForest | 0.692 | 0.780 | 0.177 | 0.284 |
| MLPRegressor | 0.616 | 0.725 | 0.208 | 0.318 |
| DNN (Keras) | — | 0.649 | 0.240 | 0.359 |

> ℹ️ 全モデルが同一のテストセットで評価されています。テストセットは特徴量選択・モデル学習に一切使用されていません。

---

## クイックスタート

```bash
# 1. プロジェクトをクローン
git clone https://github.com/Nothingness-Void/Graduation-project
cd Graduation-project

# 2. 依存関係をインストール
pip install -r requirements.txt
conda install -c conda-forge rdkit

# 3. データ統合 + 特徴量エンジニアリング + 特徴量選択 + モデリング
python 合并数据集.py              # 新旧データ統合
python 特征工程.py                # 全量 RDKit 記述子 (320 次元)
python 遗传.py                   # GA 粗篩 (320 → ~20-40, 約 20-40 分)
python 特征筛选.py                # RFECV 精篩 (~20-40 → ~8-15)
python Sklearn_AutoTune.py       # Sklearn 自動チューニング
```

---

## 評価指標

| 指標 | 式 | 説明 |
|-----|---|------|
| **R²** | 1 - SS_res/SS_tot | 決定係数（1 に近いほど良好） |
| **MAE** | mean(\|y_true - y_pred\|) | 平均絶対誤差 |
| **RMSE** | √(mean((y_true - y_pred)²)) | 二乗平均平方根誤差 |

---

## ライセンス

本プロジェクトは卒業論文プロジェクトであり、学術研究目的のみに使用されます。
