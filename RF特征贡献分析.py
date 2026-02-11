"""

随机森林特征重要性分析脚本
从训练好的模型中提取 feature_importances_，绘制排序后的特征重要性柱状图

"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# ========== 配置 ==========
MODEL_PATH = "results/fingerprint_model.pkl"        # 模型文件路径
OUTPUT_PATH = "results/RF_feature_importance.png"   # 输出图片路径

# 加载模型
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"已加载模型: {MODEL_PATH} ({model.__class__.__name__})")

# 定义特征矩阵
feature_cols = np.array(['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'MaxAbsPartialCharge1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'MaxAbsPartialCharge2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity',
                'Delta_LogP', 'Delta_TPSA', 'HB_Match', 'Delta_MolMR', 'CSP3_1', 'CSP3_2', 'Inv_T'])

# 获取特征重要性
importances = model.feature_importances_

# 获取特征重要性排序的索引
sorted_idx = np.argsort(importances)

# 打印排序后的特征重要性
print("\n============ 特征重要性排序 ============")
for i in reversed(sorted_idx):
    print(f"  {feature_cols[i]:35s}  {importances[i]:.4f}")

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_cols[sorted_idx])
plt.xlabel('Importance')
plt.title(f'Feature Importances ({model.__class__.__name__})')
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\n特征重要性图已保存至: {OUTPUT_PATH}")
plt.show()
