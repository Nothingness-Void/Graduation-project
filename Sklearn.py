import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from feature_config import ALL_FEATURE_COLS, SELECTED_FEATURE_COLS, resolve_target_col

# =========================
# 配置区
# =========================
DATA_PATH = "data/molecular_features.xlsx"
MODEL_BUNDLE_PATH = "results/sklearn_model_bundle.pkl"
LEGACY_MODEL_PATH = "results/fingerprint_model.pkl"  # 兼容旧脚本
SUMMARY_PATH = "results/sklearn_search_summary.csv"

# 可选: "selected" (16特征) / "all" (20特征)
FEATURE_MODE = "selected"

# 训练/搜索参数
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 40


def choose_features(mode: str):
    """按配置选择特征集合。"""
    if mode == "all":
        return ALL_FEATURE_COLS
    if mode == "selected":
        return SELECTED_FEATURE_COLS
    raise ValueError("FEATURE_MODE 仅支持 'all' 或 'selected'")


def build_search_configs():
    """构造每个模型对应的 Pipeline 与搜索空间。"""
    return [
        {
            "name": "Ridge",
            "pipeline": Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge())]
            ),
            "space": {
                "model__alpha": Real(1e-4, 1e4, prior="log-uniform"),
                "model__max_iter": Integer(1000, 20000),
            },
        },
        {
            "name": "Lasso",
            "pipeline": Pipeline(
                [("scaler", StandardScaler()), ("model", Lasso(random_state=RANDOM_STATE))]
            ),
            "space": {
                "model__alpha": Real(1e-4, 1e2, prior="log-uniform"),
                "model__max_iter": Integer(2000, 30000),
            },
        },
        {
            "name": "SVR",
            "pipeline": Pipeline(
                [("scaler", StandardScaler()), ("model", SVR())]
            ),
            "space": {
                "model__kernel": Categorical(["rbf", "linear", "poly"]),
                "model__C": Real(1e-2, 1e3, prior="log-uniform"),
                "model__gamma": Real(1e-4, 1e1, prior="log-uniform"),
                "model__epsilon": Real(1e-3, 0.5, prior="log-uniform"),
            },
        },
        {
            "name": "RandomForest",
            # 树模型不依赖标准化，直接使用原始特征
            "pipeline": Pipeline(
                [("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]
            ),
            "space": {
                "model__n_estimators": Integer(100, 800),
                "model__max_depth": Integer(3, 20),
                "model__min_samples_leaf": Integer(1, 8),
                "model__max_features": Categorical(["sqrt", "log2", None]),
            },
        },
        {
            "name": "GradientBoosting",
            "pipeline": Pipeline(
                [("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
            ),
            "space": {
                "model__n_estimators": Integer(80, 600),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "model__max_depth": Integer(2, 8),
                "model__subsample": Real(0.6, 1.0, prior="uniform"),
                "model__min_samples_leaf": Integer(1, 8),
            },
        },
    ]


def main():
    # 1) 读取数据与特征
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)
    feature_cols = choose_features(FEATURE_MODE)
    X = data[feature_cols]
    y = data[target_col].values

    # 2) 留出独立测试集（只在最后评估用）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"特征模式: {FEATURE_MODE}, 特征数: {len(feature_cols)}")
    print(f"样本划分: train={len(y_train)}, test={len(y_test)}")

    # 3) 在训练集上做 BayesSearchCV 选模
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_configs = build_search_configs()
    rows = []

    for cfg in tqdm(search_configs, desc="正在建模"):
        name = cfg["name"]
        search = BayesSearchCV(
            estimator=cfg["pipeline"],
            search_spaces=cfg["space"],
            n_iter=N_ITER,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)

        # 4) 用独立测试集评估（不参与调参）
        y_pred = search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        row = {
            "Model": name,
            "Best parameter": search.best_params_,
            "CV Best R2": search.best_score_,
            "Test R2": r2,
            "Test MAE": mae,
            "Test RMSE": rmse,
            "Best estimator": search.best_estimator_,
        }
        rows.append(row)

        print(
            f"\n{name}: CV Best R2={search.best_score_:.4f}, "
            f"Test R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

    # 5) 按 CV 分数选择最终模型，避免“看测试集选模”
    results = pd.DataFrame(rows)
    best_idx = results["CV Best R2"].astype(float).idxmax()
    best_row = results.loc[best_idx]
    best_model = best_row["Best estimator"]

    # 6) 保存可复现产物：模型 + 特征配置 + 目标列
    bundle = {
        "model": best_model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "feature_mode": FEATURE_MODE,
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    # 兼容旧脚本：仍保存纯模型文件
    with open(LEGACY_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    # 清理后再导出汇总（去掉不可序列化的大对象列）
    export_cols = ["Model", "Best parameter", "CV Best R2", "Test R2", "Test MAE", "Test RMSE"]
    results[export_cols].to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n==================== 结果汇总 ====================")
    print(results[export_cols].to_string(index=False))
    print(f"\n最终模型: {best_row['Model']}")
    print(f"按 CV 选择的最优参数: {best_row['Best parameter']}")
    print(f"对应 Test R2: {best_row['Test R2']:.4f}")
    print(f"\n模型包已保存: {MODEL_BUNDLE_PATH}")
    print(f"兼容模型已保存: {LEGACY_MODEL_PATH}")
    print(f"搜索汇总已保存: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
