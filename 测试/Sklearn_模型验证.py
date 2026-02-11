import pandas as pd
import pickle
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler  # 用于标准化数据

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from feature_config import SELECTED_FEATURE_COLS


# 读取数据文件
data = pd.read_excel(ROOT_DIR / 'data/molecular_features.xlsx')  # 从统一特征文件中读取数据

#定义特征矩阵
feature_cols = SELECTED_FEATURE_COLS



# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[feature_cols],
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
