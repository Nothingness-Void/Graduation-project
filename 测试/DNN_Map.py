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
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

# 创建一个函数，该函数返回你想要优化的模型
def create_model(hidden_layer_sizes=(32, 16), activation='relu', optimizer='adam', learning_rate=0.001):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], activation=activation, input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dense(hidden_layer_sizes[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(1))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mse')

    return model

# 定义参数网格
param_grid = {
    'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
    'learning_rate': [0.001, 0.0001]
}

# 创建网格搜索对象
grid = GridSearchCV(KerasRegressor(build_fn=create_model), param_grid, cv=5, scoring='neg_mean_squared_error')

# 在训练集上进行网格搜索
grid.fit(X_train, y_train)

# 获取最优的参数和分数
print("最优参数：", grid.best_params_)
print("最优分数：", grid.best_score_)

# 使用最优模型进行预测
y_pred = grid.best_estimator_.predict(X_test)

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
history = grid.best_estimator_.model.history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()