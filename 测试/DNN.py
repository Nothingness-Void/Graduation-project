# 导入所需的 Python 包
import pandas as pd
import numpy as np
import os, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# ======== 可复现性 ========
SEED = 42
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 读取数据文件
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵
featere_cols = ['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']

# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[featere_cols]], axis=1)

# 定义目标参数
y = data['χ-result'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# 标准化数据
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

##### 数据处理部分↑ ##### 分割线 ##### 模型训练部分↓ #####

# ======== 优化后的 DNN 模型 ========
# 改进点：
#   1. 使用 keras.Input 替代 input_shape（消除 Keras 3 警告）
#   2. 每层都加 BatchNorm + Dropout 防止过拟合
#   3. 使用 L2 正则化（不仅限于最后一层）
#   4. 使用 Huber Loss（比 MAE 更平滑，对异常值鲁棒）
#   5. ModelCheckpoint 自动保存验证集上表现最好的权重

inputs = keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.25)(x)

x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Dense(16)(x)
x = layers.ReLU()(x)

outputs = layers.Dense(1, activation='linear')(x)
model = keras.Model(inputs, outputs)

# 编译模型 - 使用 Huber Loss（比 MAE 更平滑收敛，比 MSE 对异常值更鲁棒）
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='huber',
    metrics=['mae']
)

model.summary()

# 创建回调
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,                # 增加耐心，给模型更多机会收敛
    restore_best_weights=True   # 自动恢复最佳权重
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)
checkpoint = ModelCheckpoint(
    'best_dnn_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# 预测
y_pred = model.predict(X_test)

# 计算并打印模型的评估指标
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'\n============ 模型评估结果 ============')
print(f'R²值为：{r2:.4f}')
print(f'MAE(平均绝对误差)值为：{mae:.4f}')
print(f'RMSE(均方根误差)值为：{rmse_val:.4f}')

model.save('DNN.h5')

# 绘制训练误差和验证误差
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss')
axes[0].set_ylabel('Huber Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# 预测 vs 真实值散点图
axes[1].scatter(y_test, y_pred, alpha=0.6, s=20)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (理想)')
axes[1].set_title(f'Predicted vs Actual (R²={r2:.4f})')
axes[1].set_xlabel('实际 χ')
axes[1].set_ylabel('预测 χ')
axes[1].legend()

plt.tight_layout()
plt.savefig('dnn_training_history.png', dpi=150)
plt.show()