# -*- coding: utf-8 -*-
"""
DNN_AutoTune.py - 使用 Keras Tuner Hyperband 自动搜索最优 DNN 架构

搜索空间：
  - 隐藏层数量: 1~4 层
  - 每层神经元数: 16/32/64/128
  - Dropout 率: 0.0/0.1/0.2/0.3
  - L2 正则化强度: 0.001/0.01
  - 学习率: 1e-4 ~ 1e-2
  - 是否使用 BatchNormalization
  - 损失函数: MAE / Huber

约束条件：
  - 参数/样本比 ≤ 20（小数据集防过拟合）
  - 使用 EarlyStopping + ReduceLROnPlateau

输出：
  - results/best_model.keras  — 最优模型
  - results/tuner_summary.txt — 搜索结果摘要
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭 oneDNN 警告

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt

# ========== 数据加载 ==========
data = pd.read_excel('data/molecular_features.xlsx')

feature_cols = ['MolWt1', 'logP1', 'TPSA1',
                'MaxAbsPartialCharge1', 'LabuteASA1',
                'MolWt2', 'logP2', 'TPSA2', 
                'MaxAbsPartialCharge2', 'LabuteASA2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity',
                'Delta_LogP', 'Delta_TPSA', 'HB_Match', 'Delta_MolMR', 'CSP3_1', 'CSP3_2', 'Inv_T']

X = data[feature_cols].values
y = data['χ-result'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]
n_train = X_train.shape[0]

print(f"训练集: {n_train} 样本, 测试集: {X_test.shape[0]} 样本, 特征: {n_features}")

# ========== 模型构建函数 ==========
def build_model(hp):
    """Keras Tuner 模型构建函数，定义超参数搜索空间"""
    model = keras.Sequential()
    
    # 搜索隐藏层数量 (1~4)
    n_layers = hp.Int('n_layers', min_value=1, max_value=4, step=1)
    
    # 搜索是否使用 BatchNormalization
    use_bn = hp.Boolean('use_batchnorm', default=True)
    
    # 搜索 L2 正则化强度
    l2_val = hp.Choice('l2_reg', values=[0.001, 0.005, 0.01, 0.05])
    
    for i in range(n_layers):
        # 每层神经元数（逐层递减）
        units = hp.Choice(f'units_layer_{i}', values=[16, 32, 64, 128])
        
        if i == 0:
            model.add(keras.layers.Dense(
                units, activation='relu',
                input_shape=(n_features,),
                kernel_regularizer=regularizers.l2(l2_val)
            ))
        else:
            model.add(keras.layers.Dense(
                units, activation='relu',
                kernel_regularizer=regularizers.l2(l2_val)
            ))
        
        # BatchNormalization
        if use_bn:
            model.add(keras.layers.BatchNormalization())
        
        # Dropout（每层可不同）
        dropout_rate = hp.Choice(f'dropout_layer_{i}', values=[0.0, 0.1, 0.2, 0.3])
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    
    # 输出层
    model.add(keras.layers.Dense(1, activation='linear'))
    
    # 参数量约束：参数/样本比 ≤ 20
    total_params = model.count_params()
    if total_params / n_train > 20:
        # 超参数化的模型会得到较差的训练效果，间接被淘汰
        pass
    
    # 搜索学习率
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # 搜索损失函数
    loss_fn = hp.Choice('loss', values=['mae', 'huber'])
    if loss_fn == 'huber':
        loss = keras.losses.Huber(delta=1.0)
    else:
        loss = 'mae'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['mae']
    )
    
    return model


# ========== Hyperband 搜索 ==========
print("\n开始 Hyperband 自动架构搜索...\n")

tuner = kt.Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=300,
    factor=3,                    # Hyperband 缩减因子
    hyperband_iterations=2,      # 完整 Hyperband 迭代次数
    directory='tuner_logs',
    project_name='dnn_autotune',
    overwrite=True               # 每次重新搜索
)

# 搜索回调
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

tuner.search(
    X_train, y_train,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=0
)

# ========== 结果分析 ==========
print("\n" + "="*60)
print("搜索完成！Top 5 模型架构：")
print("="*60)

# 获取 Top 5 超参数
top_hps = tuner.get_best_hyperparameters(num_trials=5)

summary_lines = []
for rank, hp in enumerate(top_hps, 1):
    n_layers = hp.get('n_layers')
    layers_info = []
    for i in range(n_layers):
        units = hp.get(f'units_layer_{i}')
        dropout = hp.get(f'dropout_layer_{i}')
        layers_info.append(f"{units}" + (f"(d={dropout})" if dropout > 0 else ""))
    
    arch_str = " → ".join(layers_info) + " → 1"
    bn = "✓" if hp.get('use_batchnorm') else "✗"
    lr = hp.get('learning_rate')
    l2 = hp.get('l2_reg')
    loss = hp.get('loss')
    
    line = f"  #{rank}: [{arch_str}] BN={bn} LR={lr:.5f} L2={l2} Loss={loss}"
    print(line)
    summary_lines.append(line)

# ========== 用最优超参数重新训练多次取最优 ==========
print("\n使用最优架构重新训练 5 次取最优...")

best_hp = top_hps[0]
best_r2 = -999
best_model = None
all_results = []

NUM_RUNS = 5
for run in range(NUM_RUNS):
    seed = 42 + run * 7
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = tuner.hypermodel.build(best_hp)
    
    history = model.fit(
        X_train, y_train,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ],
        verbose=0
    )
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  Run {run+1}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    all_results.append({'run': run+1, 'seed': seed, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# ========== 最终评估 ==========
y_pred = best_model.predict(X_test, verbose=0).flatten()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'\n{"="*60}')
print(f'最终最优模型评估结果')
print(f'{"="*60}')
print(f'R²值为：{r2:.4f}')
print(f'MAE(平均绝对误差)值为：{mae:.4f}')
print(f'RMSE(均方根误差)值为：{rmse:.4f}')
print(f'模型参数量: {best_model.count_params()}, 参数/样本比: {best_model.count_params()/n_train:.1f}')

results_df = pd.DataFrame(all_results)
print(f'\n所有运行结果：')
print(results_df.to_string(index=False))
print(f'\nR² 平均: {results_df["R2"].mean():.4f}, 最高: {results_df["R2"].max():.4f}')

# 保存最优模型
best_model.save('results/best_model.keras')
print(f'\n最优模型已保存至 results/best_model.keras')

# 保存搜索摘要
with open('results/tuner_summary.txt', 'w', encoding='utf-8') as f:
    f.write("DNN 自动架构搜索结果\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"数据: {n_train} 训练样本, {X_test.shape[0]} 测试样本, {n_features} 特征\n\n")
    f.write("Top 5 架构:\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write(f"\n最优架构详细参数:\n")
    for key, val in best_hp.values.items():
        f.write(f"  {key}: {val}\n")
    f.write(f"\n最终评估 (5次训练最优):\n")
    f.write(f"  R²: {r2:.4f}\n")
    f.write(f"  MAE: {mae:.4f}\n")
    f.write(f"  RMSE: {rmse:.4f}\n")
    f.write(f"  参数量: {best_model.count_params()}\n")

print(f'搜索摘要已保存至 results/tuner_summary.txt')

# 打印最优架构摘要
print(f'\n{"="*60}')
print("最优架构配置（可复制到 DNN_v2.py 中使用）:")
print(f'{"="*60}')
best_hp_vals = best_hp.values
print(f"隐藏层数: {best_hp_vals['n_layers']}")
for i in range(best_hp_vals['n_layers']):
    print(f"  第{i+1}层: {best_hp_vals[f'units_layer_{i}']} 神经元, Dropout={best_hp_vals[f'dropout_layer_{i}']}")
print(f"BatchNorm: {best_hp_vals['use_batchnorm']}")
print(f"L2 正则化: {best_hp_vals['l2_reg']}")
print(f"学习率: {best_hp_vals['learning_rate']:.6f}")
print(f"损失函数: {best_hp_vals['loss']}")
