import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# 加载最优的随机森林模型 (假设已保存为 fingerprint_model.pkl)
with open("fingerprint_model.pkl", "rb") as f:
    model = pickle.load(f)

# 定义特征矩阵
feature_cols = np.array(['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)'])

# 获取特征重要性
importances = model.feature_importances_

# 获取特征重要性排序的索引
sorted_idx = np.argsort(importances)

# 绘制特征重要性图
plt.figure(figsize=(12, 7))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_cols[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()