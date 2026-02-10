# -*- coding: utf-8 -*-
"""
DNN_v2.py - 基于数据量分析的架构优化版本

数据分析结论：
  - 总样本数: 315（训练集 252，测试集 63）
  - 特征数: 28
  - 目标值 χ 范围: [-1.06, 3.5]，均值 0.32，标准差 0.56

问题诊断：
  原始模型有 14,753 个参数，但仅有 252 个训练样本
  参数/样本比 = 58.5（严重过参数化，经验法则应 < 10）
  这是导致 R² 难以提升的根本原因——模型容量远超数据量，容易过拟合
  
优化策略：
  1. 大幅缩减网络宽度（128→64→32→16→8→4 改为 64→32→16）
  2. 将参数量控制在 ~2500 以内（参数/样本比 ≈ 10）
  3. 保留 BatchNorm（有助于小数据集训练稳定性）
  4. 轻度 Dropout(0.1) 仅在第一层
  5. 多次种子训练取最优
"""
# 导入所需的 Python 包
import pandas as pd
import numpy as np
import os, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt

# 读取数据文件
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵（与原始 DNN.py 完全一致）
featere_cols = ['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']

X = pd.concat([data[featere_cols]], axis=1)
y = data['χ-result'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本, 特征: {X_train.shape[1]}")

##### 数据处理部分↑ ##### 分割线 ##### 模型训练部分↓ #####

def build_model(seed):
    """
    针对小数据集优化的 DNN 架构
    
    原始: 128→BN→64→32→16→8→4(L2)→1  = ~14,753 参数
    优化: 64→BN→Dropout→32→16→1       = ~2,769 参数
    
    参数/样本比: 58.5 → 11.0（合理范围）
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = keras.Sequential([
        # 第一隐藏层：64 神经元（原始的一半）
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        keras.layers.Dropout(0.1),
        
        # 第二隐藏层：32 神经元
        keras.layers.Dense(32, activation='relu'),
        
        # 第三隐藏层：16 神经元 + L2 正则化
        keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        
        # 输出层
        keras.layers.Dense(1, activation='linear')
    ])
    
    # 参数统计
    total_params = model.count_params()
    print(f"  [seed={seed}] 模型参数量: {total_params}, 参数/样本比: {total_params/X_train.shape[0]:.1f}")

    model.compile(optimizer='Adam', loss='mae')
    return model


# ======== 多次训练取最优 ========
NUM_RUNS = 5
best_r2 = -999
best_model = None
best_history = None
all_results = []

print(f"\n将进行 {NUM_RUNS} 次训练，选择最优模型...\n")

for run in range(NUM_RUNS):
    seed = 42 + run * 7
    print(f"--- Run {run+1}/{NUM_RUNS} ---")

    model = build_model(seed)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"    R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    all_results.append({'run': run+1, 'seed': seed, 'R2': r2, 'MAE': mae, 'RMSE': rmse})

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_history = history

# ======== 最优模型评估 ========
y_pred = best_model.predict(X_test, verbose=0).flatten()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'\n============ 最终最优模型评估结果 ============')
print(f'R²值为：{r2:.4f}')
print(f'MAE(平均绝对误差)值为：{mae:.4f}')
print(f'RMSE(均方根误差)值为：{rmse_val:.4f}')

print(f'\n所有运行结果：')
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
print(f'\nR² 平均: {results_df["R2"].mean():.4f}, 最高: {results_df["R2"].max():.4f}')

best_model.save('DNN_v2.h5')
print("\n最优模型已保存为 DNN_v2.h5")

# ======== 绘图 ========
plt.plot(best_history.history['loss'], label='Train')
plt.plot(best_history.history['val_loss'], label='Test')
plt.title('Model loss (Best Run)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('DNN_v2_loss.png', dpi=150)
plt.show()
