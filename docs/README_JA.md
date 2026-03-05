<p align="center">
  <a href="../README.md">简体中文</a> ·
  <a href="README_EN.md">English</a> ·
  <a href="README_JA.md">日本語</a>
</p>

# 分子記述子に基づく Huggins パラメータ（chi）の QSAR 予測モデル

> この版は AI 翻訳です。表現に軽微な不自然さが含まれる場合があります。
>
> 本プロジェクトは、**QSAR（定量的構造活性相関）** 手法を用い、分子記述子と機械学習 / 深層学習モデルを利用して、高分子-溶媒系の **Huggins パラメータ（χ）** を予測します。

---

## 📋 目次

- [プロジェクト概要](#プロジェクト概要)
- [プロジェクト構成](#プロジェクト構成)
- [全体フロー概要（Step 1-6）](#全体フロー概要step-1-6)
- [モデリング段階（Step 5）](#モデリング段階step-5)
- [検証・分析段階（Step 6）](#検証分析段階step-6)
- [データファイル説明](#データファイル説明)
- [モデル性能ベンチマーク](#モデル性能ベンチマーク)
- [実験アーカイブ](#実験アーカイブ)
- [代表的な出力図](#代表的な出力図)
- [クイックスタート](#クイックスタート)
- [評価指標](#評価指標)

---

## プロジェクト概要

**Huggins パラメータ（χ）** は、高分子-溶媒相互作用を記述する重要な熱力学パラメータであり、混合系における溶媒と高分子間の親和性を反映します。

本プロジェクトの核心的なワークフローは以下の通りです：

1. 元の文献データから化合物名を抽出し、**SMILES** 分子構造表現に変換します。
2. 複数のソースからのデータセットを統合します（旧データ 323件 + 新データ 1586件、文献間の chi 矛盾をクリーニング・解決して **1815件** を取得）。
3. **RDKit** を利用して、全 **約210個** の2D分子記述子 + 指紋類似度 + 相互作用特徴量を自動計算し、**332次元の特徴量行列** を生成します。
4. **遺伝的アルゴリズム（GA）** を使用して、332次元の中から最適な特徴量サブセットを選択します。
5. 最適な特徴量に基づき、**AutoTune** を使用して ML / DNN モデルの自動ハイパーパラメータ最適化と学習を行います。

---

## プロジェクト構成

```
Graduation-project/
│
├── 获取SMILES.py              # Step 1: 化合物名 → SMILES
├── 数据处理部分代码.py          # Step 2: χ 式の解析 + 温度展開
├── 合并数据集.py               # Step 2.5: 旧データと新データの統合
├── 特征工程.py                 # Step 3: 全量 RDKit 記述子抽出 (332次元)
├── 遗传.py                    # Step 4a: GA 粗選別（性能版、RF evaluator）
├── 遗传_ElasticNet.py         # Step 4a: GA 粗選別（物理版、ElasticNet evaluator）
├── 特征筛选.py                 # Step 4b: RFECV 精選別
├── feature_config.py           # 特徴量設定センター (選択された特徴量列の統一管理)
│
├── DNN_AutoTune.py            # Step 5a: DNN Hyperband 自動チューニング
├── Sklearn_AutoTune.py        # Step 5b: Sklearn RandomizedSearch 自動チューニング
│
├── DNN_模型验证.py             # Step 6a: DNN モデル検証
├── DNN特征贡献分析.py          # Step 6c: DNN SHAP 特徴量寄与分析
├── Y_Randomization.py         # Step 6d: Sklearn Y-Randomization 検証
├── DNN_Y_Randomization.py     # Step 6e: DNN Y-Randomization 検証
│
├── data/                      # 中間データ
│   ├── smiles_raw.csv
│   ├── smiles_cleaned.xlsx
│   ├── huggins_preprocessed.xlsx
│   ├── 43579_2022_237_MOESM1_ESM.csv  # 新規外部データセット (1586件)
│   ├── merged_dataset.csv             # 統合後データセット (1815件、矛盾解決済み)
│   ├── molecular_features.xlsx        # 332次元特徴量行列
│   └── features_optimized.xlsx        # 選別後特徴量サブセット
│
├── results/                   # モデルと結果
│   ├── best_model.keras        # DNN AutoTune 最良モデル
│   ├── best_model_preprocess.pkl # DNN 前処理器 + 最良ハイパーパラメータ
│   ├── sklearn_model_bundle.pkl # Sklearn 統一モデルバンドル
│   ├── ga_best_model.pkl      # GA 選出最良モデル
│   ├── ga_selected_features.txt     # GA 選別特徴量リスト
│   ├── ga_evolution_log.csv         # GA 進化ログ
│   ├── sklearn_tuning_summary.csv   # AutoTune 探索レポート
│   ├── train_test_split_indices.npz # 統一 train/test 分割インデックス
│   └── feature_selection.png        # 特徴量選別可視化
│
├── final_results/             # 最終成果物（中間生成物と分離）
│   ├── dnn/                   # Git は .png/.csv のみ追跡、その他はローカル生成
│   │   ├── dnn_validation_plots.png
│   │   ├── dnn_validation_results.csv
│   │   ├── dnn_feature_importance.csv
│   │   └── dnn_y_randomization.png
│   └── sklearn/               # Git は .png のみ追跡、その他はローカル生成
│       ├── sklearn_feature_importance.png
│       ├── sklearn_validation_plots.png
│       └── y_randomization.png
│
├── utils/                     # 共有ユーティリティモジュール
│   ├── data_utils.py           # load_saved_split_indices 等
│   └── plot_style.py           # 統一プロットテーマ
│
├── requirements.txt           # Python 依存リスト
├── README.md                  # 本ファイル
│
├── 模型/                      # 過去モデルアーカイブ
└── 废弃文件存档/               # アーカイブ済み旧ファイル (Sklearn.py, DNN.py 等)
```

---

## 全体フロー概要（Step 1-6）

| 段階 | 主要スクリプト | 主要出力 |
|------|--------------|----------|
| Step 1：SMILES 取得 | `获取SMILES.py` | `data/smiles_raw.csv` |
| Step 2：データ前処理 | `数据处理部分代码.py`、`合并数据集.py` | `data/huggins_preprocessed.xlsx`、`data/merged_dataset.csv` |
| Step 3：特徴量エンジニアリング | `特征工程.py` | `data/molecular_features.xlsx`（332次元） |
| Step 4：特徴量選別 | `遗传.py`（性能版 GA）、`遗传_ElasticNet.py`（物理版 GA）、`特征筛选.py` | `results/ga_selected_features.txt`、`data/features_optimized.xlsx` |
| Step 5：モデル学習とチューニング | `Sklearn_AutoTune.py`、`DNN_AutoTune.py` | `final_results/sklearn/*`、`results/best_model.keras` |
| Step 6：モデル検証と分析 | `Y_Randomization.py`、`DNN_Y_Randomization.py`、`DNN特征贡献分析.py` | `final_results/sklearn/y_randomization.*`、`final_results/dnn/dnn_y_randomization.*` |

---

## モデリング段階（Step 5）

### Step 5a: DNN Hyperband 自動チューニング

**スクリプト**: [`DNN_AutoTune.py`](../DNN_AutoTune.py)

Keras Tuner の Hyperband アルゴリズムを使用して、DNN の最適アーキテクチャ（1-3層、12-64ノード、学習率、正則化など）を探索します。

| 設定項目 | 値 |
|----------|---|
| 探索戦略 | Hyperband (Keras Tuner) |
| 探索空間 | 1-3層, 12-64ノード, L2正則化, Dropout |
| データ分割 | 60% 訓練 / 20% 検証 / 20% テスト |
| 標準化 | X と y の両方に StandardScaler を使用 |
| 再学習 | 最適アーキテクチャを異なるシードで8回再学習 |

```bash
# .venv 内の Python (Keras 3 互換) を使用する必要があります
.venv\Scripts\python.exe DNN_AutoTune.py
```

### Step 5b: Sklearn AutoTune（推奨）

**スクリプト**: [`Sklearn_AutoTune.py`](../Sklearn_AutoTune.py)

4つのモデル × 50組のパラメータ × 5分割交差検証（CV）による自動最適化：

| モデル | 探索次元 |
|--------|---------|
| GradientBoosting | loss, lr, n_estimators, depth, subsample |
| XGBRegressor | lr, n_estimators, depth, reg_alpha/lambda |
| RandomForest | n_estimators, depth, max_features |
| MLPRegressor | hidden layers, activation, alpha, lr |

実行後、以下を自動的に完了します：

1. 最適モデル探索（CV 選定）
2. テストセット検証（R²/MAE/RMSE、訓練に関与していないテストセットのみを使用）
3. 特徴量寄与分析（組み込み重要度 または permutation importance）
4. 検証可視化（実測値 vs 予測値、残差分布、モデル比較など4枚の図）
5. 最終成果物を `final_results/sklearn/` に出力

```bash
python Sklearn_AutoTune.py
```

---

## 検証・分析段階（Step 6）

### モデル検証

| スクリプト | 機能 |
|------------|------|
| [`DNN_模型验证.py`](../DNN_模型验证.py) | DNN モデルをロードし、テストセットで R²/MAE/RMSE を評価 |
| [`Sklearn_AutoTune.py`](../Sklearn_AutoTune.py) | 学習終了後、Sklearn 検証結果を自動出力 (`final_results/sklearn/sklearn_validation_results.xlsx`) |

### 特徴量寄与分析

| スクリプト | 機能 |
|------------|------|
| [`DNN特征贡献分析.py`](../DNN特征贡献分析.py) | DNN 特徴量寄与の SHAP GradientExplainer 解析 |
| [`Sklearn_AutoTune.py`](../Sklearn_AutoTune.py) | 学習終了後、Sklearn 特徴量寄与を自動出力 (`final_results/sklearn/sklearn_feature_importance.*`) |

### Y-Randomization 検証

**スクリプト**: [`Y_Randomization.py`](../Y_Randomization.py)

**機能**: Y-Scrambling 検証。y値を100回ランダムにシャッフルしてモデルを再学習し、QSAR モデルが特徴量と目的変数の関係を真に学習しているかを検証します。真のモデルの R² がランダムモデルの分布より有意に高ければ (p < 0.05)、モデルは有効です。

**出力**: `final_results/sklearn/y_randomization.png`、`y_randomization.csv`

```bash
python Y_Randomization.py
```

### DNN Y-Randomization 検証

**スクリプト**: [`DNN_Y_Randomization.py`](../DNN_Y_Randomization.py)

**機能**: 同一の train/test 分割を再利用した上で、DNN の `y_train/y_val` をランダムにシャッフルして再学習を繰り返し、真の DNN とランダム化 DNN のテストセット R² 分布と p値を比較します。

**出力**: `final_results/dnn/dnn_y_randomization.csv`、`dnn_y_randomization.png`、`dnn_y_randomization_summary.txt`

```bash
python DNN_Y_Randomization.py
```

### DNN 総合検証と特徴量寄与分析（最新 AutoTune）

**スクリプト**: [`DNN特征贡献分析.py`](../DNN特征贡献分析.py)

**機能**: `best_model.keras + best_model_preprocess.pkl` を厳密に使用し、sklearn と同様の 2×2 DNN ダッシュボード（実測値 vs 予測値、残差分布、残差 vs 予測値、特徴量寄与）と、検証明細・特徴量寄与テーブルを出力します。

**出力**: `final_results/dnn/dnn_validation_plots.png`、`dnn_validation_results.csv`、`dnn_feature_importance.csv`

```bash
python DNN特征贡献分析.py
```

> `Sklearn_模型验证.py` と `RF特征贡献分析.py` は、過去の互換性とデバッグのために `废弃文件存档/` にアーカイブされました。

---

## データファイル説明

| ファイル | 場所 | 説明 | 生成段階 | Git |
|----------|------|------|----------|-----|
| `43579_2022_237_MOESM1_ESM.csv` | `data/` | 外部データセット (1586件) | 新規入力 | ✅ |
| `smiles_raw.csv` | `data/` | SMILES 照会結果 | Step 1 | ✅ |
| `smiles_cleaned.xlsx` | `data/` | 手動クリーニング後の SMILES | 手動処理 | ✅ |
| `huggins_preprocessed.xlsx` | `data/` | 前処理済みデータ (323件) | Step 2 | ✅ |
| `merged_dataset.csv` | `data/` | 統合データセット (1815件、矛盾解決済み) | Step 2.5 | ✅ |
| `molecular_features.xlsx` | `data/` | 332次元特徴量行列 | Step 3 | ✅ |
| `features_optimized.xlsx` | `data/` | 選別後特徴量サブセット | Step 4 | ✅ |
| `ga_selected_features.txt` | `results/` | GA 選別特徴量リスト | Step 4 | — |
| `ga_evolution_log.csv` | `results/` | GA 進化ログ | Step 4 | — |
| `train_test_split_indices.npz` | `results/` | 統一 train/test 分割インデックス | Step 4 | — |
| `sklearn_model_bundle.pkl` | `results/` | Sklearn 統一モデルバンドル | Step 5 | — |
| `best_model.keras` | `results/` | DNN AutoTune 最良モデル | Step 5 | — |
| `sklearn_feature_importance.png` | `final_results/sklearn/` | Sklearn 特徴量寄与図 | Step 5 | ✅ |
| `sklearn_validation_plots.png` | `final_results/sklearn/` | Sklearn 検証可視化 (4サブプロット) | Step 5 | ✅ |
| `y_randomization.png` | `final_results/sklearn/` | Y-Randomization R² 分布図 | Step 6 | ✅ |
| `dnn_validation_plots.png` | `final_results/dnn/` | DNN 総合検証図（4 サブプロット） | Step 6 | ✅ |
| `dnn_validation_results.csv` | `final_results/dnn/` | DNN テスト予測・残差明細 | Step 6 | ✅ |
| `dnn_feature_importance.csv` | `final_results/dnn/` | DNN 特徴量寄与（SHAP/フォールバック） | Step 6 | ✅ |
| `dnn_y_randomization.png` | `final_results/dnn/` | DNN Y-Randomization R² 分布図 | Step 6 | ✅ |

> ℹ️ `results/` はデフォルトでは Git 追跡対象外であり、`.gitkeep` と歴史互換用の図 `dnn_shap_analysis.png` のみ保持しています。`final_results/` 下では `.png` / `.csv` のみ Git 追跡され、その他（`.pkl`、`.txt`、`.xlsx`）はローカル産物です。

---

## モデル性能ベンチマーク

> 以下は、現在のワークスペース主線の結果です（`pre_physics`）：1815 サンプル、最終 9 特徴量（統一 train/test 分割）

| モデル | CV Val R² | Test R² | Test MAE | Test RMSE |
|--------|----------|---------|---------|---------|
| **GradientBoosting** | **0.739** | **0.852** | **0.145** | **0.250** |
| XGBRegressor | 0.725 | 0.827 | 0.165 | 0.271 |
| RandomForestRegressor | 0.699 | 0.822 | 0.162 | 0.275 |
| MLPRegressor | 0.597 | 0.748 | 0.227 | 0.327 |
| DNN (AutoTune, best run) | — | 0.764 | 0.208 | 0.316 |

> ℹ️ すべてのモデルは同一のテストセットで評価されており、テストセットは特徴量選択やモデル学習には一切関与していません。
> ℹ️ DNN 行は、AutoTune 最適アーキテクチャの8回再学習のうち最良の回の結果です（CV 平均ではありません）。
> ℹ️ デフォルト表示は 9 特徴量の `pre_physics` 解釈可能ベースラインです。他の5組の比較実験は別途アーカイブされています。

---

## 実験アーカイブ

- 六方向の結果とコードスナップショット: [`实验存档/20260302_six_way_snapshots/README.md`](../实验存档/20260302_six_way_snapshots/README.md)
- 特徴量選択探索ログ: [`实验存档/EXPLORATION_LOG.md`](../实验存档/EXPLORATION_LOG.md)

---

## 代表的な出力図

### Sklearn: 特徴量寄与

![Sklearn Feature Importance](../final_results/sklearn/sklearn_feature_importance.png)

### Sklearn: 検証可視化（4 サブプロット）

![Sklearn Validation Plots](../final_results/sklearn/sklearn_validation_plots.png)

### Sklearn: Y-Randomization 分布

![Sklearn Y-Randomization](../final_results/sklearn/y_randomization.png)

### DNN: Y-Randomization 分布

![DNN Y-Randomization](../final_results/dnn/dnn_y_randomization.png)

### DNN: 総合検証ダッシュボード（4 サブプロット）

![DNN Validation Plots](../final_results/dnn/dnn_validation_plots.png)

---

## クイックスタート

```bash
# 1. プロジェクトをクローン
git clone https://github.com/Nothingness-Void/Graduation-project
cd Graduation-project

# 2. 依存関係のインストール
pip install -r requirements.txt
conda install -c conda-forge rdkit

# 3. データ統合 + 特徴量エンジニアリング + 二段階特徴量選択 + モデリング
python 合并数据集.py              # 旧データと新データの統合
python 特征工程.py                # 全量 RDKit 記述子 (332次元)
python 遗传_ElasticNet.py        # 物理版 GA 粗選別 (332→~20 特徴量, ElasticNet, 約 10-20 分) [推奨]
# python 遗传.py                 # 性能版 GA 粗選別 (332→~30 特徴量, RF, 約 20-40 分) [代替]
python 特征筛选.py                # RFECV 精選別 (23 → 9, 最終特徴量セット)
python Sklearn_AutoTune.py       # Sklearn 自動チューニング
python DNN_AutoTune.py           # DNN Hyperband 自動チューニング
python Y_Randomization.py        # Sklearn Y-Randomization 検証（オプション）
python DNN_Y_Randomization.py    # DNN Y-Randomization 検証（オプション）

# または: data/molecular_features.xlsx が既にある場合、Step 4 から開始
python 遗传_ElasticNet.py        # GA 粗選別
python 特征筛选.py                # RFECV 精選別 (→ 9 特徴量)
python Sklearn_AutoTune.py
python DNN_AutoTune.py
```

---

## 評価指標

| 指標 | 式 | 説明 |
|------|----|------|
| **R²** | 1 - SS_res/SS_tot | 決定係数。1に近いほど良い |
| **MAE** | mean(\|y_true - y_pred\|) | 平均絶対誤差 |
| **RMSE** | √(mean((y_true - y_pred)²)) | 二乗平均平方根誤差 |

---

## License

本プロジェクトは卒業設計プロジェクトであり、学術研究目的でのみ使用されます。
