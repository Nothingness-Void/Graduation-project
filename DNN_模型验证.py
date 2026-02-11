"""

DNN 模型验证脚本
加载训练好的 DNN 模型，在数据上进行预测，并输出评估指标

"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ========== 配置 ==========
MODEL_PATH = "results/DNN.h5"               # 模型文件路径（可替换为任意 .h5 / .keras 模型）
DATA_PATH = "data/molecular_features.xlsx"           # 验证数据文件路径
OUTPUT_PATH = "results/DNN_validation_results.xlsx"  # 输出文件路径

# 加载模型
model = keras.models.load_model(MODEL_PATH, compile=False)
print(f"已加载模型: {MODEL_PATH}")

# 加载验证数据
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

# 获取特征和目标参数
X_val = data[feature_cols]
y_val = data["χ-result"].values

# 对验证数据进行标准化
scaler_X = StandardScaler()
X_val_scaled = scaler_X.fit_transform(X_val)

# 进行预测
y_pred = model.predict(X_val_scaled).flatten()

# 计算评估指标
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f'\n============ 模型验证结果 ============')
print(f'模型文件: {MODEL_PATH}')
print(f'验证样本数: {len(y_val)}')
print(f'R²值为：{r2:.4f}')
print(f'MAE(平均绝对误差)值为：{mae:.4f}')
print(f'RMSE(均方根误差)值为：{rmse:.4f}')

# 将预测结果添加到 DataFrame
data["Predicted χ-result"] = y_pred
data["Residual"] = y_val - y_pred

# 将 DataFrame 输出到新的 Excel 文件
data.to_excel(OUTPUT_PATH, index=False)
print(f'\n验证结果已保存至: {OUTPUT_PATH}')
