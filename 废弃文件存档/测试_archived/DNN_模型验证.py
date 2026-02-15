import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col

# 加载模型
model = keras.models.load_model("DNN_0.706.h5")

# 加载验证数据 
data = pd.read_excel(ROOT_DIR / 'data/molecular_features.xlsx')

#定义特征矩阵
feature_cols = SELECTED_FEATURE_COLS
target_col = resolve_target_col(data.columns)



# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[feature_cols],
# data[fingerprints]
], axis=1)

# 获取特征和目标参数
X_val = data[feature_cols]
y_val = data[target_col].values

# 对验证数据进行标准化
scaler_X = StandardScaler()
X_val = scaler_X.fit_transform(X_val)

# 进行预测
y_pred = model.predict(X_val)

# 将预测结果添加到 DataFrame
data["Predicted χ-result"] = y_pred

# 将 DataFrame 输出到新的 Excel 文件
data.to_excel("DNN_validation_results.xlsx", index=False)
