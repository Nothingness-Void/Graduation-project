<p align="center">
  <a href="README.md">简体中文</a> ·
  <a href="docs/README_EN.md">English</a> ·
  <a href="docs/README_JA.md">日本語</a>
</p>

# 基于分子描述符的哈金斯参数（Huggins Parameter）QSAR 预测模型

> 本项目通过 **QSAR（定量构效关系）** 方法，利用分子描述符和机器学习 / 深度学习模型预测聚合物-溶剂体系的 **Huggins 参数（χ）**。

---

## 📋 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [完整运行流程](#完整运行流程)
  - [Step 1：获取 SMILES 分子表示](#step-1获取-smiles-分子表示)
  - [Step 2：数据预处理](#step-2数据预处理)
  - [Step 2.5：数据合并](#step-25数据合并)
  - [Step 3：特征工程](#step-3特征工程)
  - [Step 4：特征选择](#step-4特征选择)
  - [Step 5：模型训练与自动调参](#step-5模型训练与自动调参)
  - [Step 6：模型验证与分析](#step-6模型验证与分析)
- [数据文件说明](#数据文件说明)
- [模型性能基准](#模型性能基准)
- [依赖列表](#依赖列表)

---

## 项目简介

**Huggins 参数（χ）** 是描述聚合物-溶剂相互作用的关键热力学参数，其值反映了混合体系中溶剂与聚合物之间的亲和性。

本项目的核心思路是：

1. 从原始文献数据中提取化合物名称，转换为 **SMILES** 分子结构表示
2. 合并多来源数据集（旧数据 323 条 + 新数据 1586 条 = **1893 条**）
3. 利用 **RDKit** 自动计算全部 **~210 个** 2D 分子描述符 + 指纹相似度 + 交互特征，生成 **320 维特征矩阵**
4. 使用 **遗传算法（GA）** 从 320 维中选出最优特征子集
5. 基于最优特征，使用 **AutoTune** 自动超参数优化训练 ML / DNN 模型

---

## 项目结构

```
Graduation-project/
│
├── 获取SMILES.py              # Step 1: 化合物名称 → SMILES
├── 数据处理部分代码.py          # Step 2: χ 表达式解析 + 温度裂变
├── 合并数据集.py               # Step 2.5: 合并旧数据与新数据
├── 特征工程.py                 # Step 3: 全量 RDKit 描述符提取 (320 维)
├── 特征筛选.py                 # Step 4a: RFECV 特征筛选
├── 遗传.py                    # Step 4b: 遗传算法 (GA) 特征选择
├── feature_config.py           # 特征配置中心 (统一管理选中的特征列)
│
├── DNN.py                     # Step 5a: DNN 深度神经网络建模
├── DNN_AutoTune.py            # Step 5b: DNN Hyperband 自动调参
├── Sklearn.py                 # Step 5c: Sklearn 贝叶斯优化建模
├── Sklearn_AutoTune.py        # Step 5d: Sklearn 随机搜索自动调参
│
├── DNN_模型验证.py             # Step 6a: DNN 模型验证
├── DNN特征贡献分析.py          # Step 6c: DNN SHAP 特征贡献分析
├── Y_Randomization.py         # Step 6d: Sklearn Y-Randomization 验证
├── DNN_Y_Randomization.py     # Step 6e: DNN Y-Randomization 验证
│
├── Huggins.xlsx               # 原始数据：化合物名称 + 哈金斯参数
│
├── data/                      # 中间过程数据
│   ├── smiles_raw.csv
│   ├── smiles_cleaned.xlsx
│   ├── huggins_preprocessed.xlsx
│   ├── 43579_2022_237_MOESM1_ESM.csv  # 新增外部数据集 (1586 条)
│   ├── merged_dataset.csv             # 合并后数据集 (1893 条)
│   ├── molecular_features.xlsx        # 320 维特征矩阵
│   └── features_optimized.xlsx        # 筛选后特征子集
│
├── results/                   # 模型与结果
│   ├── dnn_model.keras         # DNN 模型
│   ├── dnn_preprocess.pkl      # DNN 预处理器
│   ├── sklearn_model_bundle.pkl # Sklearn 统一模型包
│   ├── ga_best_model.pkl      # GA 选出的最优模型
│   ├── ga_selected_features.txt     # GA 选中的特征列表
│   ├── ga_evolution_log.csv         # GA 进化日志
│   ├── sklearn_tuning_summary.csv   # AutoTune 寻优报告
│   ├── train_test_split_indices.npz # 统一 train/test 划分索引
│   ├── feature_selection.png        # 特征筛选可视化
│   └── dnn_loss.png                 # 训练损失曲线
│
├── final_results/             # 最终交付结果（与中间体分离）
│   ├── dnn/
│   │   ├── dnn_y_randomization.csv
│   │   ├── dnn_y_randomization.png
│   │   └── dnn_y_randomization_summary.txt
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
├── requirements.txt           # Python 依赖清单
├── README.md                  # 本文件
│
├── 测试/                      # 实验性脚本
├── 模型/                      # 历史模型存档
├── 参考/                      # 参考代码
└── 废弃文件存档/               # 已归档的废弃文件
```

---

## 环境配置

### 前提条件

- Python 3.8+
- pip 包管理器

### 安装依赖

```bash
pip install -r requirements.txt
conda install -c conda-forge rdkit  # rdkit 需要通过 conda 安装
```

### 主要依赖

| 库 | 用途 |
|---|---|
| `pandas` / `numpy` | 数据处理与科学计算 |
| `rdkit` | 分子描述符计算、分子指纹生成 |
| `scikit-learn` | 传统机器学习模型与数据预处理 |
| `scikit-optimize` | 贝叶斯超参数优化（BayesSearchCV） |
| `xgboost` | XGBoost 回归模型 |
| `deap` | 遗传算法特征选择 |
| `tensorflow` / `keras` | 深度神经网络 (DNN) |
| `keras-tuner` | DNN Hyperband 自动调参 |
| `shap` | 模型可解释性分析（SHAP 值） |
| `joblib` | 模型序列化 |
| `matplotlib` | 数据可视化 |
| `requests` / `tqdm` | 网络请求 / 进度条显示 |

---

## 完整运行流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                          完 整 流 程 图                              │
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
│  Step 2.5: 合并数据集.py ◄─── 新数据 (ESM.csv)                      │
│       │                                                             │
│       ▼                                                             │
│  Step 3: 特征工程.py → 320 维全量 RDKit 描述符                       │
│       │                                                             │
│       ▼                                                             │
│  Step 4a: 遗传.py (GA 粗筛: 320 → ~20-40)                           │
│       │                                                             │
│       ▼                                                             │
│  Step 4b: 特征筛选.py (RFECV 精筛: ~20-40 → ~8-15)                  │
│       │                                                             │
│       ├─────────────────────┐                                       │
│       ▼                     ▼                                       │
│  Step 5a: Sklearn       Step 5b: DNN                                │
│  (Sklearn_AutoTune.py)  (DNN.py / DNN_AutoTune.py)                  │
│       │                     │                                       │
│       ▼                     ▼                                       │
│  Step 6: 模型验证 + 特征贡献分析                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Step 1：获取 SMILES 分子表示

**脚本**: [`获取SMILES.py`](获取SMILES.py)

**功能**: 将 `Huggins.xlsx` 中的化合物名称通过 PubChem / NCI API 转换为 SMILES 分子结构字符串。

```bash
python 获取SMILES.py
```

> ⚠️ 需要网络连接以访问 PubChem 和 NCI 数据库

---

### Step 2：数据预处理

**脚本**: [`数据处理部分代码.py`](数据处理部分代码.py)

**功能**: 处理 χ 表达式中的温度依赖项（如 `0.43+34.7T`），温度裂变（20-50°C），异常值过滤（`-1 < χ < 5`）。

```bash
python 数据处理部分代码.py
```

---

### Step 2.5：数据合并

**脚本**: [`合并数据集.py`](合并数据集.py)

**功能**: 将旧数据（`huggins_preprocessed.xlsx`，323 条）与新外部数据（`43579_2022_237_MOESM1_ESM.csv`，1586 条）合并为统一格式，去重后生成 **1893 条** 的合并数据集。

**数据流**: 旧数据 + 新数据 → `data/merged_dataset.csv`

**统一列格式**: `Polymer, Solvent, Polymer_SMILES, Solvent_SMILES, chi, temperature, source`

```bash
python 合并数据集.py
```

---

### Step 3：特征工程

**脚本**: [`特征工程.py`](特征工程.py)

**功能**: 使用 RDKit 的 `CalcMolDescriptors()` 自动提取全部 **~210 个 2D 分子描述符**，对聚合物和溶剂分别计算后拼接，再补充指纹相似度和交互特征。

**数据流**: `data/merged_dataset.csv` → `data/molecular_features.xlsx`

| 特征类别 | 数量 | 说明 |
|---------|------|------|
| 聚合物描述符 (后缀 `_1`) | ~148 | MolWt, LogP, TPSA, 碎片计数, 拓扑指标等 |
| 溶剂描述符 (后缀 `_2`) | ~155 | 同上 |
| 指纹相似度 | 3 | Avalon, Morgan, Topological |
| 交互特征 | 14 | Delta_LogP, Delta_TPSA, HB_Match, Inv_T 等 |
| **总计** | **~320** | 经清洗后 (去除高缺失 + 常量列) |

**特殊处理**: 聚合物 SMILES 中的 `[*]` 连接点标记会被替换为 `[H]`，确保 RDKit 正常解析。

```bash
python 特征工程.py
```

---

### Step 4：两阶段特征选择

采用 **GA 粗筛 → RFECV 精筛** 两阶段策略，从 320 维中逐步筛选最优特征子集：

```
320 维 ──GA粗筛──→ ~20-40 维 ──RFECV精筛──→ ~8-15 维 ──→ 建模
```

#### Step 4a：遗传算法 (GA) 粗筛

**脚本**: [`遗传.py`](遗传.py)

**功能**: 使用 DEAP 遗传算法从 ~320 维特征中全局搜索最优特征子集。GA 能探索特征间的非线性组合效应，适合高维粗筛。

| 参数 | 值 | 说明 |
|------|---|------|
| 种群大小 | 100 | 每代 100 个候选方案 |
| 最大代数 | 60 | 上限（通常早停） |
| 早停 | 12 代无改善 | 自动停止 |
| CV 折数 | 3 | 平衡速度与精度 |
| 评估器 | RF(n=100, depth=8) | 轻量快速 |
| 特征数约束 | [5, 40] | 控制模型复杂度 |

**输出**: `results/ga_selected_features.txt`、`results/ga_evolution_log.csv`、`results/train_test_split_indices.npz`，自动更新 `feature_config.py`

> ℹ️ GA 会创建并保存 train/test 划分索引，下游所有脚本自动复用同一划分，确保测试集完全隔离。

```bash
python 遗传.py    # 约 20-40 分钟
```

#### Step 4b：RFECV 精筛

**脚本**: [`特征筛选.py`](特征筛选.py)

**功能**: 从 GA 选出的 ~20-40 个特征中，使用 RFECV 逐个淘汰冗余特征，精确定位最优子集。自动从 `feature_config.py` 读取 GA 预选结果。

> ⚠️ 必须先运行 `遗传.py`，否则脚本会报错提示。会自动加载 GA 保存的 train/test 切分索引，仅在训练集上做筛选。

**输出**: 自动更新 `feature_config.py` 和 `data/features_optimized.xlsx`

```bash
python 特征筛选.py
```

#### 统一特征管理

**脚本**: [`feature_config.py`](feature_config.py)

特征选择结果统一存储在此文件中，定义了 `SELECTED_FEATURE_COLS`（选中特征），供下游训练和验证脚本使用。

---

### Step 5：模型训练与自动调参

#### Step 5a：DNN 深度神经网络

**脚本**: [`DNN.py`](DNN.py)

| 配置项 | 值 |
|--------|------|
| 网络结构 | 48 → BN → Dropout(0.15) → 24 → BN → Dropout(0.1) → 12(L2) → 1 |
| 损失函数 | Huber |
| 训练策略 | 5 个随机种子多次训练，选最优 |
| 数据划分 | 60% 训练 / 20% 验证 / 20% 测试 |
| 标准化 | X 和 y 均使用 StandardScaler |

```bash
# 需要使用 .venv 中的 Python (Keras 3 兼容)
.venv\Scripts\python.exe DNN.py
```

#### Step 5b：DNN Hyperband 自动调参

**脚本**: [`DNN_AutoTune.py`](DNN_AutoTune.py)

使用 Keras Tuner 的 Hyperband 算法搜索 DNN 最优架构（1-3 层、12-64 节点、学习率、正则化等）。

```bash
.venv\Scripts\python.exe DNN_AutoTune.py
```

#### Step 5c：Sklearn 传统机器学习

**脚本**: [`Sklearn.py`](Sklearn.py)

批量训练多种 Sklearn 回归模型，使用 BayesSearchCV 搜索最优参数。

#### Step 5d：Sklearn AutoTune（推荐）

**脚本**: [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py)

4 个模型 × 50 组参数 × 5 折交叉验证自动寻优：

| 模型 | 搜索维度 |
|------|---------|
| GradientBoosting | loss, lr, n_estimators, depth, subsample |
| XGBRegressor | lr, n_estimators, depth, reg_alpha/lambda |
| RandomForest | n_estimators, depth, max_features |
| MLPRegressor | hidden layers, activation, alpha, lr |

运行后会自动完成：

1. 最优模型搜索（CV 选模）
2. 测试集验证（R²/MAE/RMSE，仅用未参与训练的测试集）
3. 特征贡献分析（内置重要性或 permutation importance）
4. 验证可视化（Actual vs Predicted、残差分布、模型对比等 4 张图）
5. 将最终交付文件输出到 `final_results/sklearn/`

```bash
python Sklearn_AutoTune.py
```

---

### Step 6：模型验证与分析

#### 模型验证

| 脚本 | 功能 |
|------|------|
| [`DNN_模型验证.py`](DNN_模型验证.py) | 加载 DNN 模型，在全量数据上评估 R²/MAE/RMSE |
| [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py) | 训练结束后自动输出 Sklearn 验证结果（`final_results/sklearn/sklearn_validation_results.xlsx`） |

#### 特征贡献分析

| 脚本 | 功能 |
|------|------|
| [`DNN特征贡献分析.py`](DNN特征贡献分析.py) | SHAP GradientExplainer 分析 DNN 特征贡献 |
| [`Sklearn_AutoTune.py`](Sklearn_AutoTune.py) | 训练结束后自动输出 Sklearn 特征贡献（`final_results/sklearn/sklearn_feature_importance.*`） |

#### Y-Randomization 验证

**脚本**: [`Y_Randomization.py`](Y_Randomization.py)

**功能**: Y-Scrambling 验证，通过 100 次随机打乱 y 值并重训模型，验证 QSAR 模型是否真正学到了特征与目标值的关系。如果真实模型 R² 远高于随机模型分布 (p < 0.05)，则模型有效。

**输出**: `final_results/sklearn/y_randomization.png`、`y_randomization.csv`

```bash
python Y_Randomization.py
```

#### DNN Y-Randomization 验证

**脚本**: [`DNN_Y_Randomization.py`](DNN_Y_Randomization.py)

**功能**: 在复用同一 train/test 划分的前提下，对 DNN 的 `y_train/y_val` 进行随机打乱并重复重训，对比真实 DNN 与随机化 DNN 的测试集 R² 分布与 p-value。

**输出**: `final_results/dnn/dnn_y_randomization.csv`、`dnn_y_randomization.png`、`dnn_y_randomization_summary.txt`

```bash
python DNN_Y_Randomization.py
```

> `Sklearn_模型验证.py` 与 `RF特征贡献分析.py` 已归档至 `废弃文件存档/`，用于历史兼容与排错。

---

## 数据文件说明

| 文件 | 位置 | 描述 | 产生阶段 |
|------|------|------|----------|
| `Huggins.xlsx` | 根目录 | 原始数据 | 输入 |
| `43579_2022_237_MOESM1_ESM.csv` | `data/` | 外部数据集 (1586 条) | 新增输入 |
| `smiles_raw.csv` | `data/` | SMILES 查询结果 | Step 1 |
| `smiles_cleaned.xlsx` | `data/` | 手动清洗后的 SMILES | 手动处理 |
| `huggins_preprocessed.xlsx` | `data/` | 预处理数据 (323 条) | Step 2 |
| `merged_dataset.csv` | `data/` | 合并数据集 (1893 条) | Step 2.5 |
| `molecular_features.xlsx` | `data/` | 320 维特征矩阵 | Step 3 |
| `features_optimized.xlsx` | `data/` | 筛选后特征子集 | Step 4 |
| `ga_selected_features.txt` | `results/` | GA 选中的特征列表 | Step 4b |
| `ga_evolution_log.csv` | `results/` | GA 进化日志 | Step 4b |
| `sklearn_model_bundle.pkl` | `results/` | Sklearn 统一模型包 | Step 5 |
| `dnn_model.keras` | `results/` | DNN 模型 | Step 5 |
| `train_test_split_indices.npz` | `results/` | 统一 train/test 划分索引 | Step 4a |
| `sklearn_final_report.txt` | `final_results/sklearn/` | Sklearn 最终报告 | Step 5d |
| `sklearn_validation_results.xlsx` | `final_results/sklearn/` | Sklearn 验证结果明细 | Step 5d |
| `sklearn_feature_importance.png` | `final_results/sklearn/` | Sklearn 特征贡献图 | Step 5d |
| `sklearn_validation_plots.png` | `final_results/sklearn/` | Sklearn 验证可视化 (4 张子图) | Step 5d |
| `y_randomization.png` | `final_results/sklearn/` | Y-Randomization R² 分布图 | Step 6 |
| `y_randomization.csv` | `final_results/sklearn/` | Y-Randomization 详细数据 | Step 6 |
| `dnn_y_randomization.png` | `final_results/dnn/` | DNN Y-Randomization R² 分布图 | Step 6 |
| `dnn_y_randomization.csv` | `final_results/dnn/` | DNN Y-Randomization 详细数据 | Step 6 |
| `dnn_y_randomization_summary.txt` | `final_results/dnn/` | DNN Y-Randomization 统计摘要 | Step 6 |

---

## 模型性能基准

> 以下为合并数据集 (1886 样本, 6 特征 RFECV) 上的 AutoTune 结果

| 模型 | CV Val R² | Test R² | Test MAE | Test RMSE |
|------|----------|---------|---------|---------|
| **GradientBoosting** | **0.749** | **0.812** | 0.156 | 0.263 |
| XGBRegressor | 0.726 | 0.799 | 0.150 | 0.271 |
| RandomForest | 0.692 | 0.780 | 0.177 | 0.284 |
| MLPRegressor | 0.616 | 0.725 | 0.208 | 0.318 |
| DNN (Keras) | — | 0.649 | 0.240 | 0.359 |

> ℹ️ 所有模型均在相同的测试集上评估，测试集不参与特征选择或模型训练。
> 💡 使用 GA 从 320 维特征中选择最优子集后，性能有望进一步提升。

---

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/Nothingness-Void/Graduation-project
cd Graduation-project

# 2. 安装依赖
pip install -r requirements.txt
conda install -c conda-forge rdkit

# 3. 数据合并 + 特征工程 + 两阶段特征选择 + 建模
python 合并数据集.py              # 合并旧数据与新数据
python 特征工程.py                # 全量 RDKit 描述符 (320 维)
python 遗传.py                   # GA 粗筛 (320 → ~20-40, 约 20-40 min)
python 特征筛选.py                # RFECV 精筛 (~20-40 → ~8-15)
python Sklearn_AutoTune.py       # Sklearn 自动调参
python Y_Randomization.py        # Sklearn Y-Randomization 验证（可选）
python DNN_Y_Randomization.py    # DNN Y-Randomization 验证（可选）

# 或: 如果已有 data/molecular_features.xlsx, 从 Step 4 开始
python 遗传.py
python Sklearn_AutoTune.py
```

---

## 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **R²** | 1 - SS_res/SS_tot | 决定系数，越接近 1 越好 |
| **MAE** | mean(\|y_true - y_pred\|) | 平均绝对误差 |
| **RMSE** | √(mean((y_true - y_pred)²)) | 均方根误差 |

---

## License

本项目为毕业设计项目，仅供学术研究使用。
