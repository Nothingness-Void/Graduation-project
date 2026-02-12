# -*- coding: utf-8 -*-
"""
Sklearn_AutoTune.py - 使用 RandomizedSearchCV 自动搜索最优回归模型

包含模型：
1. MLPRegressor (神经网络)
2. RandomForestRegressor (随机森林)
3. XGBRegressor (梯度提升树)
4. GradientBoostingRegressor (梯度提升)
5. SVR (支持向量机)

改进:
- 使用 feature_config 统一特征管理
- 搜索迭代次数增加到 50
- 增加 GradientBoosting 模型
- 扩大 XGBoost / MLP 搜索空间
- 评分目标使用 R2
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
import joblib
import os

from feature_config import ALL_FEATURE_COLS, SELECTED_FEATURE_COLS, resolve_target_col

# 忽略警告
warnings.filterwarnings('ignore')

# 确保结果目录存在
if not os.path.exists('results'):
    os.makedirs('results')

# =========================
# 配置区
# =========================
DATA_PATH = "data/molecular_features.xlsx"
MODEL_PATH = "results/best_model_sklearn.pkl"
SUMMARY_PATH = "results/sklearn_tuning_summary.txt"

# 可选: "selected" (16特征) / "all" (20特征)
FEATURE_MODE = "selected"

# 训练参数
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER = 50  # 每个模型随机尝试 50 组参数


def choose_features(mode: str):
    """按配置选择特征集合。"""
    if mode == "all":
        return ALL_FEATURE_COLS
    if mode == "selected":
        return SELECTED_FEATURE_COLS
    raise ValueError("FEATURE_MODE 仅支持 'all' 或 'selected'")


# ========== 1. 数据加载 ==========
print("正在加载数据...")
data = pd.read_excel(DATA_PATH)
target_col = resolve_target_col(data.columns)
feature_cols = choose_features(FEATURE_MODE)

# 确保特征存在
missing_cols = [c for c in feature_cols if c not in data.columns]
if missing_cols:
    print(f"警告：以下特征缺失，将跳过: {missing_cols}")
    feature_cols = [c for c in feature_cols if c in data.columns]

X = data[feature_cols].values
y = data[target_col].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"特征模式: {FEATURE_MODE}, 特征数: {len(feature_cols)}")
print(f"数据准备完成：训练集 {X_train.shape[0]} 样本，测试集 {X_test.shape[0]} 样本")

# ========== 2. 定义搜索空间（使用 Pipeline 封装标准化） ==========

models_param_grid = {
    'MLPRegressor': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(max_iter=5000, early_stopping=True,
                                   validation_fraction=0.15, n_iter_no_change=30)),
        ]),
        'params': {
            'model__hidden_layer_sizes': [
                (64,), (128,), (256,),
                (64, 32), (128, 64), (256, 128),
                (128, 64, 32), (256, 128, 64),
                (128, 64, 32, 16),
            ],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam'],
            'model__alpha': [1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.05],
            'model__learning_rate_init': [0.0005, 0.001, 0.005, 0.01],
            'model__batch_size': [16, 32, 64],
        }
    },
    'XGBRegressor': {
        'pipeline': Pipeline([
            ('model', XGBRegressor(n_jobs=-1, objective='reg:squarederror', random_state=RANDOM_STATE)),
        ]),
        'params': {
            'model__n_estimators': [100, 200, 300, 500, 800],
            'model__max_depth': [3, 4, 5, 6, 7, 9],
            'model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
            'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
            'model__reg_alpha': [0, 0.01, 0.1, 0.5, 1, 5],
            'model__reg_lambda': [0.1, 0.5, 1, 5, 10],
            'model__min_child_weight': [1, 3, 5, 7],
            'model__gamma': [0, 0.01, 0.1, 0.5, 1],
        }
    },
    'RandomForestRegressor': {
        'pipeline': Pipeline([
            ('model', RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        'params': {
            'model__n_estimators': [200, 300, 500, 800, 1000],
            'model__max_depth': [None, 5, 10, 15, 20, 30],
            'model__min_samples_split': [2, 3, 5, 8, 10],
            'model__min_samples_leaf': [1, 2, 3, 4],
            'model__max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        }
    },
    'GradientBoosting': {
        'pipeline': Pipeline([
            ('model', GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]),
        'params': {
            'model__n_estimators': [100, 200, 300, 500, 800],
            'model__max_depth': [2, 3, 4, 5, 6, 8],
            'model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
            'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'model__min_samples_leaf': [1, 2, 3, 5, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', None],
            'model__loss': ['squared_error', 'huber'],
        }
    },
    'SVR': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR()),
        ]),
        'params': {
            'model__kernel': ['rbf', 'linear', 'poly'],
            'model__C': [0.01, 0.1, 1, 10, 50, 100, 500],
            'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'model__epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
        }
    }
}

# ========== 3. 自动搜索 ==========
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
best_overall_model = None
best_overall_r2 = -float('inf')
best_overall_name = ""

print(f"\n开始自动寻优 (每个模型 {N_ITER} 组参数, 5 折交叉验证)...")

for name, config in models_param_grid.items():
    print(f"\n正在调优: {name} ...")

    search = RandomizedSearchCV(
        config['pipeline'],
        config['params'],
        n_iter=N_ITER,
        cv=kf,
        scoring='r2',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # 评估
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 提取 CV 训练/验证 R2 以检测过拟合
    cv_train_r2 = search.cv_results_['mean_train_score'][search.best_index_]
    cv_val_r2 = search.best_score_

    print(f"  最优参数: {search.best_params_}")
    print(f"  CV 训练 R2: {cv_train_r2:.4f}, CV 验证 R2: {cv_val_r2:.4f}")
    print(f"  测试集 R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    results.append({
        'Model': name,
        'Best Params': str(search.best_params_),
        'CV Train R2': cv_train_r2,
        'CV Val R2': cv_val_r2,
        'Test R2': r2,
        'Test MAE': mae,
        'Test RMSE': rmse,
        'Object': best_model
    })

    if cv_val_r2 > best_overall_r2:
        best_overall_r2 = cv_val_r2
        best_overall_model = best_model
        best_overall_name = name

# ========== 4. 总结与保存 ==========
print("\n" + "="*60)
print("所有模型调优结果汇总")
print("="*60)
results_df = pd.DataFrame(results).sort_values(by='CV Val R2', ascending=False)
display_cols = ['Model', 'CV Train R2', 'CV Val R2', 'Test R2', 'Test MAE', 'Test RMSE']
print(results_df[display_cols].to_string(index=False))

print(f"\n最佳模型 (按 CV R2 选择): {best_overall_name}")
print(f"最佳 CV R2: {best_overall_r2:.4f}")

# 保存
if best_overall_model is not None:
    joblib.dump(best_overall_model, MODEL_PATH)
    print(f"最佳模型已保存至: {MODEL_PATH}")

    # 保存摘要
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write("Sklearn 自动寻优结果汇总\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"特征模式: {FEATURE_MODE}, 特征数: {len(feature_cols)}\n")
        f.write(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}\n")
        f.write(f"每模型迭代次数: {N_ITER}\n\n")
        f.write(results_df[display_cols].to_string(index=False))
        f.write(f"\n\n最佳模型: {best_overall_name}\n")
        f.write(f"最佳参数:\n")
        best_row = results_df.iloc[0]
        f.write(str(best_row['Best Params']))

    print(f"搜索摘要已保存至: {SUMMARY_PATH}")
