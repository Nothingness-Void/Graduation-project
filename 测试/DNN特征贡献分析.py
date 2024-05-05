import shap
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 加载模型
model = load_model("DNN_0.706.h5")

# 加载数据
data = pd.read_excel("计算结果.xlsx")  # 将文件名替换为实际文件名

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

# 获取特征和目标参数
X_val = data[featere_cols]
y_val = data["χ-result"].values

# 对验证数据进行标准化
scaler_X = StandardScaler()
X_val = scaler_X.fit_transform(X_val)

# 初始化JS解释器
explainer = shap.GradientExplainer(model, X_val)

# 计算SHAP值
shap_values = explainer.shap_values(X_val)

# 计算每个特征的平均绝对SHAP值
mean_shap_values = np.mean(np.abs(shap_values[0]), axis=0)

# 获取特征重要性排序的索引
sorted_idx = np.argsort(mean_shap_values)

# 绘制特征重要性图
plt.figure(figsize=(10, 7))
plt.barh(range(len(sorted_idx)), mean_shap_values[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(feature_cols)[sorted_idx])
plt.xlabel('SHAP Value (average impact on model output)')
plt.title('Feature importances')
plt.draw()
plt.show()