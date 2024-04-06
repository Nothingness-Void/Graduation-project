# 导入所需的 Python 包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.layers import BatchNormalization


# 读取数据文件
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵
X = data[['Similarity', 'MolWt1', 'logP1', 'TPSA1', 'MolWt2', 'logP2', 'TPSA2']].values

# 定义目标参数
y = data['χ-result'].values

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

# 导入 KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# 定义模型构建函数
def create_model(optimizer='adam', init_mode='uniform', activation='relu', dropout_rate=0.0, weight_constraint=0):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=activation, kernel_initializer=init_mode, kernel_constraint=tf.keras.constraints.max_norm(weight_constraint), input_shape=(X_train.shape[1],),kernel_regularizer=regularizers.l1(0.01)),#输入层
        BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(32, activation=activation, kernel_regularizer=regularizers.l2(0.01)),#隐藏层5
        keras.layers.Dense(16, activation=activation, kernel_regularizer=regularizers.l2(0.01)),#隐藏层6
        keras.layers.Dense(8, activation=activation, kernel_regularizer=regularizers.l2(0.01)),#隐藏层7
        keras.layers.Dense(1)  # 输出层
    ])
    model.compile(optimizer=optimizer, loss= 'mse')
    return model

# 创建 KerasRegressor 对象
model = KerasRegressor(build_fn=create_model, verbose=0)

# 定义网格搜索参数
param_grid = {
    'batch_size': [10, 20, 40, 60, 80, 100],
    'epochs': [10, 50, 100],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'weight_constraint': [1, 2, 3, 4, 5]
}

# 创建 GridSearchCV 对象
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, cv=3)
grid_result = grid.fit(X_train, y_train)

# 输出最优的模型参数
print("最优参数: %f 使用 %s" % (grid_result.best_score_, grid_result.best_params_))

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