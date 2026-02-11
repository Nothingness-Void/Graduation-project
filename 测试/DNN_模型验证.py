import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 加载模型
model = keras.models.load_model("DNN_0.706.h5")

# 加载验证数据 
data = pd.read_excel('计算结果.xlsx')

#定义特征矩阵
featere_cols = ['MolWt1', 'logP1', 'TPSA1',
                'dipole1', 'LabuteASA1',
                'MolWt2', 'logP2', 'TPSA2', 
                'dipole2', 'LabuteASA2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']



# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[featere_cols],
# data[fingerprints]
], axis=1)

# 获取特征和目标参数
X_val = data[featere_cols]
y_val = data["χ-result"].values

# 对验证数据进行标准化
scaler_X = StandardScaler()
X_val = scaler_X.fit_transform(X_val)

# 进行预测
y_pred = model.predict(X_val)

# 将预测结果添加到 DataFrame
data["Predicted χ-result"] = y_pred

# 将 DataFrame 输出到新的 Excel 文件
data.to_excel("DNN_validation_results.xlsx", index=False)