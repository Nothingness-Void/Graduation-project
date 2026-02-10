# 导入所需的 Python 包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras import backend as K


# 读取数据文件
data = pd.read_excel('data/molecular_features.xlsx')

# 识别目标列（处理 Excel 列名乱码）
target_col = 'χ-result'
if target_col not in data.columns:
    result_candidates = [c for c in data.columns if 'result' in str(c).lower()]
    if result_candidates:
        target_col = result_candidates[0]
    else:
        raise KeyError("未找到目标列：χ-result（或包含 result 的列名）")

# 从现有相似度列中选择与目标相关性最高的一列
similarity_cols = [c for c in ['Avalon Similarity', 'Morgan Similarity', 'Topological Similarity'] if c in data.columns]
if similarity_cols:
    corrs = data[similarity_cols].corrwith(data[target_col]).abs()
    best_similarity_col = corrs.idxmax()
    data['Similarity'] = data[best_similarity_col]
else:
    best_similarity_col = None

# 定义特征矩阵（优先使用与之前一致的 7 个特征）
use_baseline_features = True
if use_baseline_features:
    if 'Similarity' not in data.columns:
        raise KeyError("未找到相似度列：Similarity / Avalon Similarity / Morgan Similarity / Topological Similarity")
    feature_cols = ['Similarity', 'MolWt1', 'logP1', 'TPSA1', 'MolWt2', 'logP2', 'TPSA2']
else:
    feature_cols = ['MolWt1', 'logP1', 'TPSA1',
                    'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                    'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                    'MolWt2', 'logP2', 'TPSA2', 
                    'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                    'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                    'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']

# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[feature_cols],
# data[fingerprints]
], axis=1)

# 定义目标参数
y = data[target_col].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 标准化数据
scaler_X = StandardScaler() 
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))
##### 数据处理部分↑ ##### 分割线 ##### 模型训练部分↓ #####

# 构建 DNN 模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],),),#输入层
    BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),#隐藏层2
    keras.layers.Dense(32, activation='relu'),#隐藏层3
    #keras.layers.Dense(32, activation='relu'),#隐藏层4
    #keras.layers.Dense(32, activation='relu'),#隐藏层5
    keras.layers.Dense(16, activation='relu'),#隐藏层6
    keras.layers.Dense(8, activation='relu'),#隐藏层7
    keras.layers.Dense(4, activation='relu',kernel_regularizer=regularizers.l2(0.01)),#隐藏层5
    #keras.layers.Dense(2, activation='relu',kernel_regularizer=regularizers.l2(0.01)),#隐藏层6
    keras.layers.Dense(1, activation='linear') #输出层
])

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# 编译模型
#model.compile(optimizer='adam', loss=rmse)  # 使用 Adam 优化器和平均绝对误差作为损失函数
model.compile(optimizer='Adam', loss= 'mae')  # 使用 Adam 优化器和平均绝对误差作为损失函数
#model.compile(optimizer='Adam', loss= 'mape')


# 创建早停回调
early_stopping = EarlyStopping(monitor='val_loss', patience=15)

# 创建学习率衰减回调
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# 训练与评估模型
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# 预测
y_pred = model.predict(X_test)

# 反标准化
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# 计算并打印模型的R^2值
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'R^2值为：{r2}')
print(f'MAE(平均绝对误差)值为：{mae}')
print(f'RMSE(均方根误差)值为：{rmse}')
#print(f'MAPE(平均绝对百分比误差)值为：{mape}')

model.save('results/DNN.h5')

# 绘制训练误差和验证误差
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()