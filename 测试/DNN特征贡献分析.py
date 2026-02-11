import shap
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from feature_config import SELECTED_FEATURE_COLS

# 加载模型
model = load_model("DNN_0.706.h5")

# 加载数据
data = pd.read_excel(ROOT_DIR / "data/molecular_features.xlsx")  # 将文件名替换为实际文件名

#定义特征矩阵
feature_cols = SELECTED_FEATURE_COLS

# 获取特征和目标参数
X_val = data[feature_cols]

# 对验证数据进行标准化
scaler_X = StandardScaler()
X_val = scaler_X.fit_transform(X_val)

# 将X_val转换回DataFrame
X_val = pd.DataFrame(X_val, columns=feature_cols)

# 初始化GradientExplainer解释器
explainer = shap.GradientExplainer(model, X_val)

# 计算SHAP值
shap_values = explainer.shap_values(X_val.values)

print("SHAP values ：", shap_values)

# 选择第一个输出的 SHAP 值
shap_values_output1 = shap_values[0]

# 计算每个特征的平均绝对 SHAP 值
mean_abs_shap_values = np.mean(np.abs(shap_values_output1), axis=0)

# 计算每个特征的平均 SHAP 值
mean_shap_values = np.mean(shap_values_output1, axis=0)

# 创建一个新的图形
plt.figure(figsize=(10, 8))

# 绘制平均绝对 SHAP 值
plt.subplot(1, 2, 1)
plt.barh(feature_cols, mean_abs_shap_values)
plt.xlabel('Mean Absolute SHAP Value')
plt.title('Feature Importance (Absolute)')

# 绘制平均 SHAP 值
plt.subplot(1, 2, 2)
plt.barh(feature_cols, mean_shap_values)
plt.xlabel('Mean SHAP Value')
plt.title('Feature Importance (Signed)')

plt.tight_layout()
plt.show()
