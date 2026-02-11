# -*- coding: utf-8 -*-
"""
Sklearn_AutoTune.py - 使用 RandomizedSearchCV 自动搜索最优回归模型 (无需 TensorFlow)

包含模型：
1. MLPRegressor (神经网络)
2. RandomForestRegressor (随机森林)
3. XGBRegressor (梯度提升树)
4. SVR (支持向量机)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
import joblib
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 确保结果目录存在
if not os.path.exists('results'):
    os.makedirs('results')

# ========== 1. 数据加载 ==========
print("正在加载数据...")
try:
    data = pd.read_excel('data/features_optimized.xlsx')
except FileNotFoundError:
    print("找不到 data/features_optimized.xlsx，尝试使用 molecular_features.xlsx")
    data = pd.read_excel('data/molecular_features.xlsx')
    
try:
    from feature_config import SELECTED_FEATURE_COLS as feature_cols
except ImportError:
    print("Warning: feature_config.py not found. Using default features.")
    feature_cols = ['MolWt1', 'LabuteASA1', 'logP2', 'Delta_TPSA', 'HB_Match', 'Inv_T']

# 确保特征存在
missing_cols = [c for c in feature_cols if c not in data.columns]
if missing_cols:
    print(f"警告：以下特征缺失，将跳过: {missing_cols}")
    feature_cols = [c for c in feature_cols if c in data.columns]

X = data[feature_cols].values
y = data['χ-result'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"数据准备完成：训练集 {X_train.shape[0]} 样本，测试集 {X_test.shape[0]} 样本，特征 {X_train.shape[1]} 维")

# ========== 2. 定义搜索空间 ==========

models_param_grid = {
    'MLPRegressor': {
        'model': MLPRegressor(max_iter=5000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20),
        'params': {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.05],
            'learning_rate_init': [0.001, 0.005, 0.01]
        }
    },
    'XGBRegressor': {
        'model': XGBRegressor(n_jobs=-1, objective='reg:absoluteerror'),
        'params': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0.1, 1, 5]
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(n_jobs=-1, random_state=42),
        'params': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.5]
        }
    }
}

# ========== 3. 自动搜索 ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_overall_model = None
best_overall_r2 = -float('inf')
best_overall_name = ""

print("\n开始自动寻优...")

for name, config in models_param_grid.items():
    print(f"\n正在调优: {name} ...")
    
    search = RandomizedSearchCV(
        config['model'],
        config['params'],
        n_iter=20,  # 每个模型随机尝试 20 组参数
        cv=kf,
        scoring='neg_mean_absolute_error', # 以优化 MAE 为目标
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    # 评估
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  最优参数: {search.best_params_}")
    print(f"  测试集 R²: {r2:.4f}")
    print(f"  测试集 MAE: {mae:.4f}")
    
    results.append({
        'Model': name,
        'Best Params': str(search.best_params_),
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Object': best_model
    })
    
    if r2 > best_overall_r2:
        best_overall_r2 = r2
        best_overall_model = best_model
        best_overall_name = name

# ========== 4. 总结与保存 ==========
print("\n" + "="*60)
print("所有模型调优结果汇总")
print("="*60)
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print(results_df[['Model', 'R2', 'MAE', 'RMSE']].to_string(index=False))

print(f"\n最佳模型: {best_overall_name}")
print(f"最佳 R²: {best_overall_r2:.4f}")

# 保存
if best_overall_model is not None:
    model_path = 'results/best_model_sklearn.pkl'
    joblib.dump(best_overall_model, model_path)
    print(f"最佳模型已保存至: {model_path}")

    # 保存摘要
    with open('results/sklearn_tuning_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Sklearn 自动寻优结果汇总\n")
        f.write("="*60 + "\n")
        f.write(results_df[['Model', 'R2', 'MAE', 'RMSE']].to_string(index=False))
        f.write("\n\n最佳模型参数:\n")
        f.write(str(results_df.iloc[0]['Best Params']))
