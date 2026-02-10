"""

Sklearn 模型验证脚本
加载训练好的 Sklearn 模型，在数据上进行预测，并输出评估指标

"""
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========== 配置 ==========
MODEL_PATH = "fingerprint_model.pkl"  # 模型文件路径
DATA_PATH = "计算结果.xlsx"            # 验证数据文件路径
OUTPUT_PATH = "Sklearn_validation_results.xlsx"  # 输出文件路径

# 读取数据文件
data = pd.read_excel(DATA_PATH)

# 定义特征矩阵
feature_cols = ['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']

X = data[feature_cols]
y_val = data["χ-result"].values

# 加载模型
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"已加载模型: {MODEL_PATH} ({model.__class__.__name__})")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 对特征进行预测
y_pred = model.predict(X_scaled)

# 计算评估指标
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f'\n============ 模型验证结果 ============')
print(f'模型文件: {MODEL_PATH}')
print(f'模型类型: {model.__class__.__name__}')
print(f'验证样本数: {len(y_val)}')
print(f'R²值为：{r2:.4f}')
print(f'MAE(平均绝对误差)值为：{mae:.4f}')
print(f'RMSE(均方根误差)值为：{rmse:.4f}')

# 将预测结果保存到 Excel 文件
data["Predicted χ-result"] = y_pred
data["Residual"] = y_val - y_pred
data.to_excel(OUTPUT_PATH, index=False, encoding='utf_8_sig')
print(f'\n验证结果已保存至: {OUTPUT_PATH}')
