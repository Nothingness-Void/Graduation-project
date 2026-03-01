# Flory-Huggins 显式交互项与 MI 预筛实验存档

## 实验背景
本轮实验尝试在主线 QSAR 流程中显式加入 Flory-Huggins 理论启发的温度交互项：

- `InvT_x_DLogP`
- `InvT_x_DMolMR`
- `InvT_x_DMaxCharge`
- `InvT_x_DTPSA`
- `InvT_x_HBMatch`

同时尝试将 `mutual information` 预筛接入 `GA -> RFECV` 主流程，以降低 GA 搜索空间并提高稳定性。

## 核心结论
我们尝试加入 Flory-Huggins 理论启发的显式交互项（`InvT_x_Delta_X`），但实验证明树模型可以内部隐式构造等效交互，显式引入反而因共线性干扰了特征选择过程，导致最终模型性能整体未优于 `pre_performance` 基线。

更具体地说：

1. 在 `post_performance` 路线中，显式交互项没有稳定进入最终特征子集。
2. 在 `MI + GA` 路线中，即便对物理先验特征做强制保留，树模型性能仍未超过 `pre_performance`。
3. DNN 在 `post_performance` 上有局部提升，但整体主线模型的综合表现、流程简洁性和结果可复现性仍以 `pre_performance` 更稳妥。

## 代表性结果
| 方案 | 最终特征数 | 最佳树模型 Test R2 | DNN best R2 |
|--|--:|--:|--:|
| pre_performance | 30 | 0.8538 | 0.7936 |
| post_performance | 10 | 0.8443 | 0.8148 |
| mi_conservative | 14 | 0.8479 | 0.7464 |

## 决策
主线代码与运行产物恢复到 `pre_performance` 版本。
本目录仅作为实验存档，保留如下材料：

- 实验版 `特征工程.py`
- MI 保守版 `遗传.py`
- `遗传_ElasticNet.py`
- 四组与五组对比分析报告
- MI 保守版关键结果摘要

## 说明
若后续重新尝试该方向，建议优先做：

1. 物理先验保护下的更温和过滤式粗筛
2. 多随机种子稳定性选择
3. 单独验证显式交互项是否应在 RFECV 阶段继续强制保留
