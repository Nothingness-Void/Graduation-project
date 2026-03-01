# 特征选择探索日志

> 本文件记录了 Huggins 参数预测（χ-QSAR）项目中特征选择阶段的完整探索过程，
> 包括发现的问题、尝试的方案、负向实验结果及最终决策。

---

## 一、问题发现：温度特征（`Inv_T`）在 GA 筛选中被丢弃

### 背景

Flory-Huggins 理论给出 χ 与温度的基本关系：

$$\chi = A + \frac{B}{T}$$

因此 `Inv_T = 1000/T` 应当是预测 χ 最不可缺少的物理特征之一。

### 发现过程

在初始运行结果中，`feature_config.py` 的 `SELECTED_FEATURE_COLS` 列表中不包含任何温度相关特征。
追踪链路：

1. `特征工程.py` 正确计算了 `Inv_T` 并写入 `molecular_features.xlsx` ✅
2. `遗传.py` 的 `load_data()` 将所有列（含 `Inv_T`）作为候选特征加入 GA ✅
3. **GA 搜索阶段**：`Inv_T` 在 53 代进化中始终未被选中 ❌

### 根因分析

GA 使用 `RF(n_estimators=100, max_depth=8)` 作为适应度评估器。
χ 与 1/T 是**全局线性关系**，但 RF 是分段常数树模型，对全局线性效应的 importance 评分系统性偏低。
同时，373 K 的样本占全数据集 22%，导致温度分布的分裂信息增益被稀释。
GA 在进化第 41 代后 12 代无改善触发早停，此时种群已收敛到不含 `Inv_T` 的局部最优。

### 修复方案

在 `遗传.py` 中引入 `MANDATORY_FEATURES` 保护机制：

```python
MANDATORY_FEATURES = ["Inv_T"]  # 物理先验必选特征
```

在个体初始化、交叉、变异三个阶段均调用 `repair_individual()`，强制 `Inv_T` 对应的基因位保持为 1。

---

## 二、双轨 GA 策略：性能版 vs 物理版

### 动机

温度特征被 RF 评估器漏掉，暴露了纯数据驱动特征选择的局限：
**RF importance 偏爱树分裂友好型特征，对全局线性物理关系不敏感**。

因此引入物理解释型 GA，使用 ElasticNet（线性模型 + L1 稀疏）作为评估器，
对 χ = A + B/T 这类线性结构更敏感。

### 实验设计

| 版本 | 文件 | 评估器 | 目标 |
|------|------|--------|------|
| 性能版 | `遗传.py` | RF / XGB（可切换） | 最高预测精度 |
| 物理版 | `遗传_ElasticNet.py` | ElasticNet + StandardScaler | 物理可解释性 |

### 结果（pre 阶段，无交互项）

| 版本 | RFECV 最终特征数 | GBR Test R2 | DNN mean R2 |
|------|:---:|:---:|:---:|
| pre_performance | 30 | **0.854** | 0.761 ± 0.018 |
| pre_physics | **9** | 0.852 | 0.726 ± 0.036 |

**结论**：物理版用 9 个具有明确 Flory-Huggins 对应的特征，
实现了与 30 个数据驱动特征几乎相同的预测精度（差 0.002）。

### 物理版最终 9 特征解读

| 特征 | 物理含义 | FH 理论对应 |
|------|---------|------------|
| `Inv_T` | 1000/T | RT 项（χ ∝ 1/T） |
| `Delta_MaxAbsCharge` | 聚合物-溶剂最大电荷差 | δ_p（极性分量） |
| `Delta_MolMR` | 摩尔折射率差 | δ_d（色散分量）|
| `MinAbsPartialCharge_1` | 聚合物最小电荷 | 电荷分布均匀度 |
| `BCUT2D_MWLOW_1` | 聚合物 BCUT 描述符 | 分子连通性 |
| `HeavyAtomMolWt_2` | 溶剂重原子分子量 | 溶剂 V_s |
| `SMR_VSA5_2` | 溶剂 SMR 表面积分区 | 极性表面 |
| `VSA_EState1_1` | 聚合物电拓扑态表面积 | 电子密度分布 |
| `VSA_EState3_2` | 溶剂电拓扑态表面积 | 电子密度分布 |

---

## 三、显式交互项实验及其负面结果

### 动机

基于 Hansen 三分量溶度参数模型：

$$\chi \approx \frac{V_s}{RT}\left[(\delta_{d1}-\delta_{d2})^2 + 0.25(\delta_{p1}-\delta_{p2})^2 + 0.25(\delta_{h1}-\delta_{h2})^2\right]$$

交互项形式 `Inv_T × Delta_X` 直接对应公式中的 `(δ₁-δ₂)²/T` 结构。
理论上应能提升 ElasticNet 的线性拟合能力。

### 加入的交互项

| 特征 | 公式 | 物理含义 |
|------|------|---------|
| `InvT_x_DLogP` | (1000/T) × \|ΔLogP\| | 温度调节疏水性差异 |
| `InvT_x_DMolMR` | (1000/T) × \|ΔMolMR\| | 温度调节极化率差异 |
| `InvT_x_DMaxCharge` | (1000/T) × \|ΔMaxCharge\| | 温度调节静电作用 |
| `InvT_x_DTPSA` | (1000/T) × \|ΔTPSA\| | 温度调节极性表面积差异 |
| `InvT_x_HBMatch` | (1000/T) × HB_Match | 温度调节氢键匹配 |

### 实验结果（post 阶段，有交互项）

| 版本 | RFECV 特征数 | GBR Test R2 | DNN mean R2 |
|------|:---:|:---:|:---:|
| post_performance | 10 | 0.844 (↓0.010) | **0.797 ± 0.012** |
| post_physics | 9 | 0.829 (↓0.023) | 0.707 ± 0.029 |

**GBR 性能在加入交互项后全线下滑**。唯一例外：post_performance 的 DNN mean R2 从 0.761 升至 0.797，
但这是因为交互项的共线性触发 RFECV 将特征数从 30 缩减至 10，DNN 在更稀疏特征集上泛化更好——这是副作用而非交互项本身的功劳。

### 失败原因分析

1. **共线三角**：`Inv_T`、`Delta_X`、`InvT_x_DX` 三者高度共线，ElasticNet 的 L1 被迫选一抛二，误杀了更干净的 Delta 特征。
2. **树模型不需要显式交互**：GBR/RF/XGB 通过多层分裂可以隐式构造等效交互，显式引入是冗余信息。
3. **GA 搜索受干扰**：5 个新特征改变了特征矩阵的相关结构，导致 GA 适应度评估不稳定。

### 决策

交互项实验结果作为**负向实验证据**保留，不纳入主线。
代码恢复至 pre 阶段（移除 `特征工程.py` 中的 `InvT_x_*` 计算）。

---

## 四、MI 预筛选方案

### 动机

多种子测试揭示 GA 的不稳定性：

```
seed=42:  GA Test R2 = 0.829
seed=7:   GA Test R2 = 0.550
seed=123: GA Test R2 = 0.649
```

根本原因：搜索空间 2³²⁵ ≈ 10⁹⁸，种群=100 × 代数=60 仅评估约 6000 个体，覆盖率趋近于零。

### 方案设计

在 `遗传.py` 的 main() 中，train/test split 之后、GA 搜索之前，插入 MI 预筛层：

```
Stage 1: MI 粗筛 (确定性)  → 325 → 61 特征
Stage 2: GA 组合搜索        → 61 → N 特征
Stage 3: RFECV 精筛         → 最终特征集
```

搜索空间从 2³²⁵ 降至 2⁶¹，缩小约 10⁸⁰ 倍。

### 关键发现

MI 排名显示 `Inv_T` 排名 **#78**（MI=0.178），不在 top-60 内，
每次运行均由 `MANDATORY_FEATURES` 强制保护机制加入。
MI top-1 为 `Delta_MaxAbsCharge`（MI=0.317），与物理版 GA 结果高度吻合。

### 实验结果

| 版本 | RFECV 特征数 | GBR Test R2 | DNN mean R2 |
|------|:---:|:---:|:---:|
| mi_performance_100 | 22 | 0.845 | 0.779 ± 0.012 |
| mi_physics_100 | 15 | 0.851 | 0.729 ± 0.024 |

mi_physics_100 的 GBR R2=0.851 与 pre_physics（0.852）几乎持平，证明 MI+ElasticNet GA 路线可行，
但 DNN 稳定性未超越 post_performance。

---

## 五、六组方案总结与最终选择

### 汇总表

| 方案 | RFECV 特征 | GBR Test R2 | DNN best R2 | DNN mean R2 | 可解释性 |
|------|:---:|:---:|:---:|:---:|:---:|
| pre_performance | 30 | **0.854** | 0.794 | 0.761 | 低（共线冗余） |
| **pre_physics** | **9** | **0.852** | 0.764 | 0.726 | **高** |
| post_performance | 10 | 0.844 | **0.815** | **0.797** | 中 |
| post_physics | 9 | 0.829 | 0.742 | 0.707 | 高 |
| mi_performance_100 | 22 | 0.845 | 0.795 | 0.779 | 中 |
| mi_physics_100 | 15 | 0.851 | 0.765 | 0.729 | 较高 |

所有六组的 Y-Randomization p-value < 0.01，统计显著性无差异。

### 最终选择：pre_physics（9 特征）

**理由**：

1. GBR R2=0.852，仅低于最高值 0.002（30 个特征才达到 0.854，奥卡姆剃刀原则）
2. 9 个特征均有明确的 Flory-Huggins 理论对应，可直接与文献对话
3. 流程最简洁：ElasticNet GA 收敛比 RF GA 更稳定，结果可复现性更强
4. 负向实验（交互项、MI+GA）均有完整归档，可作为方法论探索的补充说明

---

## 附：存档目录结构

```
实验存档/
├── 20260301_flory_huggins_explicit_interactions_mi/
│   ├── README.md          ← 本轮实验背景与决策
│   ├── reports/           ← 四组/五组对比分析报告
│   └── scripts/           ← 实验版关键脚本快照
└── 20260302_six_way_snapshots/
    ├── README.md           ← 六组快照说明
    ├── six_way_details.json
    ├── six_way_summary.csv
    ├── 01_pre_performance/ ～ 06_mi_physics_100/
    │   ├── result_snapshot/
    │   ├── code_snapshot/
    │   └── manifest.json
```
