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


# 读取数据文件
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵
X = data[['Similarity', 'MolWt1', 'logP1', 'TPSA1', 'MolWt2', 'logP2', 'TPSA2']].values

# 定义目标参数
y = data['χ-result'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 SMOTE 进行数据增强 (在此处添加)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_train, y_train = smote.fit_resample(X_train, y_train)


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
    keras.layers.Dense(32, activation='ReLU', input_shape=(X_train.shape[1],),),#输入层
    BatchNormalization(),
    #keras.layers.Dense(256, activation='relu'),#隐藏层1
    #keras.layers.Dense(128, activation='relu'),#隐藏层2
    keras.layers.Dense(8, activation='ReLU',), #隐藏层3 ,L2正则化
    keras.layers.Dense(8, activation='relu',), #隐藏层4 ,L2正则化
    keras.layers.Dense(8, activation='elu'),#隐藏层4
    keras.layers.Dense(8, activation='elu'),#隐藏层5
    #keras.layers.Dense(4, activation='elu'),#隐藏层5
    #BatchNormalization(),
    #keras.layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.01)),#隐藏层5
    #keras.layers.Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.01)),#隐藏层6
    keras.layers.Dense(1, activation='linear') #输出层
])


# 编译模型

model.compile(optimizer='adam', loss= 'mse') # 使用 Adam 优化器和均方误差作为损失函数
#model.compile(optimizer='Adam', loss= 'mae')  # 使用 Adam 优化器和平均绝对误差作为损失函数
#model.compile(optimizer='Adam', loss= 'mape')


# 创建早停回调
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

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
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'R^2值为：{r2}')
print(f'MAE(平均绝对误差)值为：{mae}')
print(f'MAPE(平均绝对百分比误差)值为：{mape}')

# 绘制训练误差和验证误差
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()