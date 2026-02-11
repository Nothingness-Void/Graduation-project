"""

DNN 特征贡献分析脚本（SHAP）
使用 GradientExplainer 计算各特征对 DNN 预测的 SHAP 值，
输出特征重要性柱状图（绝对值 + 带符号）

"""
import shap
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# ========== 配置 ==========
MODEL_PATH = "results/DNN.h5"                  # DNN 模型文件路径
DATA_PATH = "data/molecular_features.xlsx"     # 特征数据文件路径
OUTPUT_PATH = "results/DNN_SHAP_analysis.png"  # 输出图片路径

# 加载模型
model = keras.models.load_model(MODEL_PATH, compile=False)
print(f"已加载模型: {MODEL_PATH}")

# 加载数据
data = pd.read_excel(DATA_PATH)

# 定义特征矩阵
feature_cols = ['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'MaxAbsPartialCharge1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'MaxAbsPartialCharge2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity',
                'Delta_LogP', 'Delta_TPSA', 'HB_Match', 'Delta_MolMR', 'CSP3_1', 'CSP3_2', 'Inv_T']

# 获取特征并标准化
X_val = data[feature_cols]
scaler_X = StandardScaler()
X_val_scaled = scaler_X.fit_transform(X_val)
X_val_df = pd.DataFrame(X_val_scaled, columns=feature_cols)

# 初始化 GradientExplainer 解释器
print("正在计算 SHAP 值...")
explainer = shap.GradientExplainer(model, X_val_df)
shap_values = explainer.shap_values(X_val_df.values)

# 选择第一个输出的 SHAP 值
shap_values_output = shap_values[0]

# 计算每个特征的平均绝对 SHAP 值
mean_abs_shap = np.mean(np.abs(shap_values_output), axis=0)

# 计算每个特征的平均 SHAP 值（带符号）
mean_shap = np.mean(shap_values_output, axis=0)

# 按绝对值从大到小排序
sorted_idx = np.argsort(mean_abs_shap)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# 左图：平均绝对 SHAP 值
axes[0].barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], align='center')
axes[0].set_yticks(range(len(sorted_idx)))
axes[0].set_yticklabels(np.array(feature_cols)[sorted_idx])
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].set_title('Feature Importance (Absolute)')

# 右图：平均 SHAP 值（带符号）
colors = ['#e74c3c' if v > 0 else '#3498db' for v in mean_shap[sorted_idx]]
axes[1].barh(range(len(sorted_idx)), mean_shap[sorted_idx], align='center', color=colors)
axes[1].set_yticks(range(len(sorted_idx)))
axes[1].set_yticklabels(np.array(feature_cols)[sorted_idx])
axes[1].set_xlabel('Mean SHAP Value')
axes[1].set_title('Feature Importance (Signed)')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\n特征贡献分析图已保存至: {OUTPUT_PATH}")
plt.show()
