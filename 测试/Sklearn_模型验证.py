import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # 用于标准化数据


# 读取数据文件
data = pd.read_excel('计算结果.xlsx')  # 从包含七个特征的文件中读取数据

#定义特征矩阵
featere_cols = ['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']



# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[featere_cols],
# data[fingerprints]
], axis=1)

# 加载模型
with open('fingerprint_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 使用训练数据拟合 StandardScaler 并进行标准化

# 对特征进行预测
y_pred = model.predict(X_scaled)  # 使用标准化后的特征进行预测

# 将预测结果保存到 result.csv 文件中
result = pd.DataFrame({'预测结果': y_pred})
result.to_excel('Sklearn_validation_results.xlsx', index=False, encoding='utf_8_sig')