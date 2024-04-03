import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # 用于标准化数据


# 读取数据文件
data = pd.read_excel('计算结果.xlsx')  # 从包含七个特征的文件中读取数据

# 定义特征矩阵
X = data[['Similarity', 'MolWt1', 'logP1', 'TPSA1', 'MolWt2', 'logP2', 'TPSA2']].values

# 将 X 转换为二维数组
X = X.reshape(-1, X.shape[1])  # X.shape[1] 表示特征数量

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
result.to_excel('模型验证.xlsx', index=False, encoding='utf_8_sig')