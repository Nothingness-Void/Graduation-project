import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# 加载最优的随机森林模型 (假设已保存为 fingerprint_model.pkl)
with open("fingerprint_model.pkl", "rb") as f:
    model = pickle.load(f)

# 定义特征矩阵
feature_cols = np.array(['MolWt1', 'logP1', 'TPSA1',
                'dipole1', 'LabuteASA1',
                'MolWt2', 'logP2', 'TPSA2', 
                'dipole2', 'LabuteASA2',
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