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
  - [Step 3：特征工程](#step-3特征工程)
  - [Step 4：模型训练](#step-4模型训练)
  - [Step 5：模型验证与分析](#step-5模型验证与分析)
- [数据文件说明](#数据文件说明)
- [实验版本（测试目录）](#实验版本测试目录)
- [依赖列表](#依赖列表)

---

## 项目简介

**Huggins 参数（χ）** 是描述聚合物-溶剂相互作用的关键热力学参数，其值反映了混合体系中溶剂与聚合物之间的亲和性。

本项目的核心思路是：

1. 从原始文献数据中提取化合物名称，转换为 **SMILES** 分子结构表示
2. 利用 **RDKit** 计算丰富的分子描述符（QSPR 描述符 + 立体描述符 + 分子指纹相似度）
3. 基于这些描述符构建机器学习（Sklearn）和深度学习（DNN）模型，预测 χ 值

---

## 项目结构

```
Graduation-project/
│
├── 获取SMILES.py              # Step 1: 通过化合物名称获取 SMILES 字符串
├── 数据处理部分代码.py          # Step 2: 处理哈金斯参数中的浮动值与温度
├── 特征工程.py                 # Step 3: 计算分子描述符与指纹相似度
├── DNN.py                     # Step 4a: DNN 深度神经网络建模
├── Sklearn.py                 # Step 4b: Sklearn 传统机器学习建模
├── DNN_模型验证.py             # Step 5a: DNN 模型验证
├── Sklearn_模型验证.py         # Step 5b: Sklearn 模型验证
├── DNN特征贡献分析.py          # Step 6a: DNN SHAP 特征贡献分析
├── RF特征贡献分析.py           # Step 6b: 随机森林特征重要性分析
│
├── Huggins.xlsx               # 原始数据：化合物名称 + 哈金斯参数
│
├── data/                      # 中间过程数据（流水线各步骤产物）
│   ├── smiles_raw.csv         # Step 1 输出：含 SMILES 的原始数据
│   ├── smiles_cleaned.xlsx    # 手动清洗后的 SMILES 数据
│   ├── huggins_preprocessed.xlsx  # Step 2 输出：温度裂变后的预处理数据
│   └── molecular_features.xlsx    # Step 3 输出：28维分子描述符特征矩阵
│
├── results/                   # 最终成果（模型 + 验证结果 + 可视化）
│   ├── DNN.h5                 # DNN 训练模型
│   ├── DNN_v2.h5 / DNN_v2.keras  # DNN 优化版模型
│   ├── fingerprint_model.pkl  # Sklearn 最优模型
│   ├── DNN_validation_results.xlsx   # DNN 验证结果
│   ├── Sklearn_validation_results.xlsx  # Sklearn 验证结果
│   ├── DNN_v2_loss.png        # 训练损失曲线图
│   ├── DNN_v2_results.png     # 预测结果散点图
│   ├── DNN_SHAP_analysis.png  # Step 6a: DNN SHAP 特征贡献分析图
│   └── RF_feature_importance.png  # Step 6b: 随机森林特征重要性图
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
```

### 主要依赖

| 库 | 用途 |
|---|---|
| `pandas` / `numpy` | 数据处理与科学计算 |
| `rdkit` | 分子描述符计算、分子指纹生成 |
| `scikit-learn` | 传统机器学习模型与数据预处理 |
| `scikit-optimize` | 贝叶斯超参数优化（BayesSearchCV） |
| `xgboost` | XGBoost 回归模型 |
| `tensorflow` / `keras` | 深度神经网络 (DNN) |
| `matplotlib` | 数据可视化 |
| `shap` | 模型可解释性分析（SHAP 值） |
| `requests` / `tqdm` | 网络请求 / 进度条显示 |

> ⚠️ **注意**：`rdkit` 需要通过 conda 安装：`conda install -c conda-forge rdkit`

---

## 完整运行流程

整个项目按以下 **5 个步骤** 依次执行，形成完整的数据处理—建模—验证流水线：

```
┌─────────────────────────────────────────────────────────────────┐
│                        完 整 流 程 图                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Huggins.xlsx (原始数据)                                        │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐                                          │
│  │ Step 1: 获取SMILES │  →  data/smiles_raw.csv                │
│  │  (获取SMILES.py)   │     (化合物名称→SMILES分子结构)          │
│  └──────────────────┘                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────────┐                                    │
│  │ Step 2: 数据预处理       │  →  data/huggins_preprocessed.xlsx │
│  │  (数据处理部分代码.py)    │     (处理χ浮动值、温度裂变20-50°C)  │
│  └────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐                                          │
│  │ Step 3: 特征工程   │  →  data/molecular_features.xlsx       │
│  │  (特征工程.py)     │     (分子描述符 + 指纹相似度 + 目标值)    │
│  └──────────────────┘                                          │
│       │                                                         │
│       ├────────────────────┐                                    │
│       ▼                    ▼                                    │
│  ┌─────────────┐   ┌──────────────┐                            │
│  │ Step 4a: DNN │   │ Step 4b: ML  │                            │
│  │  (DNN.py)    │   │ (Sklearn.py) │                            │
│  │  → results/  │   │ → results/   │                            │
│  └─────────────┘   └──────────────┘                            │
│       │                    │                                    │
│       ▼                    ▼                                    │
│  ┌──────────────────────────────────────┐                       │
│  │ Step 5: 模型验证                      │                       │
│  │  (DNN_模型验证.py / Sklearn_模型验证.py) │                      │
│  └──────────────────────────────────────┘                       │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────┐                       │
│  │ Step 6: 特征贡献分析（可解释性）        │                       │
│  │  (DNN特征贡献分析.py / RF特征贡献分析.py) │                     │
│  └──────────────────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Step 1：获取 SMILES 分子表示

**脚本**: [`获取SMILES.py`](获取SMILES.py)

**功能**: 将 `Huggins.xlsx` 中的化合物名称通过在线 API 转换为 SMILES 分子结构字符串。

**数据流**: `Huggins.xlsx` → `SMILES.csv`

**核心逻辑**:

1. 读取 `Huggins.xlsx` 中的化合物名称列（`Compound 1`、`Compound 2`）
2. 对聚合物名称进行预处理：
   - 如 `poly(styrene)` → 提取括号内 `styrene`
   - 如 `polystyrene` → 去除 `poly` 前缀
3. 依次通过两个在线数据库查询 SMILES：
   - **一次查询**: PubChem REST API
   - **二次查询**（一次失败时）: NCI CACTUS 服务
4. 使用线程池并发查询，加速处理
5. 输出结果保存为 `SMILES.csv`

```bash
python 获取SMILES.py
```

> ⚠️ 需要网络连接以访问 PubChem 和 NCI 数据库，查询之间有随机延迟以避免被限流

---

### Step 2：数据预处理

**脚本**: [`数据处理部分代码.py`](数据处理部分代码.py)

**功能**: 处理原始哈金斯参数数据中的复杂表达式（含浮动值、温度依赖项），计算出确定的 χ 值。

**数据流**: `data/smiles_raw.csv` → `data/huggins_preprocessed.xlsx`

**核心逻辑**:

1. **χ 表达式解析** (`process_chi`):
   - 处理形如 `0.43+34.7T` 的温度依赖表达式（`T` 项除以温度 K）
   - 去除误差标识（`±`）和特殊字符
   - 支持多项加减运算
2. **温度处理** (`process_T`):
   - 缺失温度 → 默认 298.15 K
   - 范围温度（如 `300-400`）→ 取平均值
3. **数据裂变** (`split_row`):
   - 固定 χ 值 → 直接计算
   - 含 `T` 的表达式 + 无温度 → 在 4 个常见实验温度下（**20°C, 30°C, 40°C, 50°C**）分别计算
   - 含 `T` 的表达式 + 温度范围 → 在 4 个分位点温度下计算
   - 含 `T` 的表达式 + 固定温度 → 在该温度下计算
4. **异常值过滤**:
   - 自动剔除 **χ < -1 或 χ > 5** 的极端离群值（防止训练失真）
   - 保证数据分布在合理的物理化学范围内，排除非聚合物体系或特殊反应体系的干扰

> **温度选取依据**: 20-50°C 区间覆盖绝大多数有机溶剂的液态稳定区和常见实验室操作温度，既保证了数据多样性，又避免了极端温度（如 0°C 或 100°C）可能违反物理前提的问题。

```bash
python 数据处理部分代码.py
```

---

### Step 3：特征工程

**脚本**: [`特征工程.py`](特征工程.py)

**功能**: 基于 SMILES 字符串计算 28 维分子描述符特征，用于后续建模。

**数据流**: `processed_and_split_Smiles.xlsx` → `计算结果.xlsx`

**核心逻辑**:

对每一对化合物（Compound 1 + Compound 2），利用 **RDKit** 计算以下三类特征：

#### 1. QSPR 描述符（每种化合物各计算一组）

| 特征 | 说明 |
|------|------|
| `MolWt` | 精确分子量 |
| `logP` | 疏水性系数（ClogP） |
| `TPSA` | 拓扑极性表面积 |
| `LabuteASA` | Labute 约化表面积 |
| `dipole` | Crippen 描述符（LogP 相关） |

#### 2. 立体描述符（需 3D 构象优化）

| 特征 | 说明 |
|------|------|
| `asphericity` | 分子不对称性 |
| `eccentricity` | 分子偏心率 |
| `inertial_shape_factor` | 惯性形状因子 |
| `NPR1` / `NPR2` | 主惯性矩比 |
| `CalcSpherocityIndex` | 球形度指数 |
| `CalcRadiusOfGyration` | 回旋半径 |

#### 3. 分子指纹相似度（化合物对之间的相似性）

| 特征 | 说明 |
|------|------|
| `Avalon Similarity` | Avalon 指纹 Tanimoto 相似度 |
| `Morgan Similarity` | Morgan (ECFP) 指纹 Tanimoto 相似度 |
| `Topological Similarity` | 拓扑扭转指纹 Tanimoto 相似度 |

加上 `Measured at T (K)` 温度和目标值 `χ-result`，共输出 **28 个特征列 + 1 个目标列**。

```bash
python 特征工程.py
```

---

### Step 4：模型训练

特征工程完成后，可通过两种路径进行建模：

#### Step 4a：DNN 深度神经网络

**脚本**: [`DNN.py`](DNN.py)

**功能**: 使用 TensorFlow/Keras 构建 DNN 回归模型预测 χ 值。

| 配置项 | 值 |
|--------|------|
| 网络结构 | 128 → BN → 64 → 32 → 16 → 8 → 4(L2) → 1 |
| 损失函数 | MAE |
| 优化器 | Adam |
| 回调 | EarlyStopping(patience=15) + ReduceLROnPlateau |
| 数据划分 | 80% 训练 / 20% 测试 |
| 标准化 | X 和 y 均使用 StandardScaler |

```bash
python DNN.py
```

**输出**: `DNN.h5`（模型文件）+ 训练损失曲线图

#### Step 4b：Sklearn 传统机器学习

**脚本**: [`Sklearn.py`](Sklearn.py)

**功能**: 批量训练多种 Sklearn 回归模型，使用 **贝叶斯超参数优化**（BayesSearchCV）搜索最优参数，自动选出最佳模型。

| 模型 | 搜索空间 |
|------|----------|
| Ridge | alpha, max_iter |
| Lasso | alpha, max_iter |
| SVR | kernel, C, gamma, max_iter |
| RandomForest | n_estimators, max_depth, max_features |
| GradientBoosting | n_estimators, max_depth, max_features |

```bash
python Sklearn.py
```

**输出**: `fingerprint_model.pkl`（最优模型 pickle 文件）

---

### Step 5：模型验证与分析

#### DNN 模型验证

**脚本**: [`DNN_模型验证.py`](DNN_模型验证.py)

**功能**: 加载训练好的 DNN 模型，在全量数据上进行预测，输出评估指标（R²、MAE、RMSE）和预测结果。

模型路径通过脚本顶部的 `MODEL_PATH` 变量配置，默认为 `DNN.h5`，可替换为任意 `.h5` / `.keras` 模型文件。

```bash
python DNN_模型验证.py
```

**输出**: `DNN_validation_results.xlsx`（含预测值和残差列）

#### Sklearn 模型验证

**脚本**: [`Sklearn_模型验证.py`](Sklearn_模型验证.py)

**功能**: 加载训练好的 Sklearn 模型，在全量数据上进行预测，输出评估指标和预测结果。

```bash
python Sklearn_模型验证.py
```

**输出**: `Sklearn_validation_results.xlsx`（含预测值和残差列）

### Step 6：特征贡献分析（模型可解释性）

#### DNN SHAP 分析

**脚本**: [`DNN特征贡献分析.py`](DNN特征贡献分析.py)

**功能**: 使用 SHAP GradientExplainer 计算各特征对 DNN 模型预测的贡献度，输出:

- 平均绝对 SHAP 值柱状图（特征重要性排序）
- 平均 SHAP 值柱状图（带正负方向，反映特征对预测的推高/拉低效应）

```bash
python DNN特征贡献分析.py
```

**输出**: `results/DNN_SHAP_analysis.png`

#### 随机森林特征重要性

**脚本**: [`RF特征贡献分析.py`](RF特征贡献分析.py)

**功能**: 从训练好的 RandomForest 模型中提取 `feature_importances_`，按重要性排序绘制柱状图，并打印特征排名。

```bash
python RF特征贡献分析.py
```

**输出**: `results/RF_feature_importance.png`

---

## 数据文件说明

| 文件 | 位置 | 描述 | 产生阶段 |
|------|------|------|----------|
| `Huggins.xlsx` | 根目录 | 原始数据，包含化合物名称、χ 参数表达式、温度 | 输入数据 |
| `smiles_raw.csv` | `data/` | 化合物名称 + SMILES 分子结构 | Step 1 输出 |
| `smiles_cleaned.xlsx` | `data/` | 手动清洗后的 SMILES 数据 | 手动处理 |
| `huggins_preprocessed.xlsx` | `data/` | 温度裂变后的预处理数据 | Step 2 输出 |
| `molecular_features.xlsx` | `data/` | 28维分子描述符特征矩阵 + 目标值 | Step 3 输出 |
| `DNN.h5` / `DNN_v2.h5` | `results/` | DNN 训练模型 | Step 4a 输出 |
| `fingerprint_model.pkl` | `results/` | Sklearn 最优模型 | Step 4b 输出 |
| `DNN_validation_results.xlsx` | `results/` | DNN 验证结果（含预测值与残差） | Step 5 输出 |
| `Sklearn_validation_results.xlsx` | `results/` | Sklearn 验证结果（含预测值与残差） | Step 5 输出 |
| `DNN_SHAP_analysis.png` | `results/` | DNN SHAP 特征贡献分析图 | Step 6a 输出 |
| `RF_feature_importance.png` | `results/` | 随机森林特征重要性图 | Step 6b 输出 |

---

## 实验版本（测试目录）

`测试/` 目录中包含了多个实验性版本，用于探索不同模型结构和优化策略：

| 脚本 | 说明 |
|------|------|
| `测试/DNN.py` | DNN 改进版：Huber Loss + 全层 BN/Dropout + L2 正则化 + ModelCheckpoint |
| `测试/DNN_v2.py` | DNN 精简版：针对小数据集优化网络宽度，多次种子训练取最优 |
| `测试/DNN_Map.py` | DNN 超参数搜索：使用 GridSearchCV + KerasRegressor 搜索层结构和学习率 |
| `测试/Sklearn.py` | Sklearn 扩展版：增加 XGBoost 和 MLPRegressor，绘制交叉验证箱线图 |
| `测试/GPR.py` | 高斯过程回归（Gaussian Process Regression）模型 |
| `测试/DNN特征贡献分析.py` | SHAP 特征重要性分析（DNN 模型可解释性） |
| `测试/RF特征贡献分析.py` | 随机森林特征重要性排序图 |

---

## 依赖列表

详见 [`requirements.txt`](requirements.txt)：

```
pandas>=1.5.0
numpy>=1.23.0
openpyxl>=3.0.0
scikit-learn>=1.1.0
scikit-optimize>=0.9.0
xgboost>=1.7.0
tensorflow>=2.10.0
matplotlib>=3.5.0
requests>=2.28.0
tqdm>=4.64.0
```

额外依赖（需 conda 安装）：

```
rdkit (conda install -c conda-forge rdkit)
shap  (pip install shap)  # 特征贡献分析所需
```

---

## 评估指标

模型使用以下指标评估预测效果：

| 指标 | 公式 | 说明 |
|------|------|------|
| **R²** | 1 - SS_res/SS_tot | 决定系数，越接近 1 越好 |
| **MAE** | mean(\|y_true - y_pred\|) | 平均绝对误差 |
| **RMSE** | √(mean((y_true - y_pred)²)) | 均方根误差 |
| **MAPE** | mean(\|y_true - y_pred\|/\|y_true\|) | 平均绝对百分比误差 |

---

## 快速开始

```bash
# 1. 克隆项目
git clone <repository-url>
cd Graduation-project

# 2. 安装依赖
pip install -r requirements.txt
conda install -c conda-forge rdkit

# 3. 按顺序执行流水线
python 获取SMILES.py            # Step 1: 获取 SMILES（需网络）
python 数据处理部分代码.py        # Step 2: 数据预处理（温度裂变 20-50°C）
python 特征工程.py               # Step 3: 特征工程
python DNN.py                   # Step 4a: DNN 建模
python Sklearn.py               # Step 4b: Sklearn 建模
python DNN_模型验证.py           # Step 5a: DNN 模型验证
python Sklearn_模型验证.py       # Step 5b: Sklearn 模型验证
python DNN特征贡献分析.py        # Step 6a: SHAP 特征贡献分析
python RF特征贡献分析.py         # Step 6b: 随机森林特征重要性
```

> 💡 如果已有 `data/molecular_features.xlsx`，可直接从 Step 4 开始运行模型训练。

---

## License

本项目为毕业设计项目，仅供学术研究使用。
