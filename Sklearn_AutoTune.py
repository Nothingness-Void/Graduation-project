# -*- coding: utf-8 -*-
"""
Sklearn_AutoTune.py
使用 RandomizedSearchCV 自动搜索最优回归模型，并保存统一模型包供验证/贡献分析复用。
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col
from utils.data_utils import load_saved_split_indices
from utils.plot_style import (
    COLORS,
    HEXBIN_CMAP,
    add_stat_box,
    apply_plot_theme,
    plot_model_comparison,
    plot_top_barh,
    style_axis,
)

warnings.filterwarnings("ignore")
apply_plot_theme()


# ========== 配置区 ==========
DATA_PATH = "data/molecular_features.xlsx"
MODEL_BUNDLE_PATH = "results/sklearn_model_bundle.pkl"
LEGACY_MODEL_PATH = "results/fingerprint_model.pkl"  # 兼容历史脚本
SUMMARY_PATH = "results/sklearn_tuning_summary.csv"
FINAL_ROOT_DIR = "final_results"
FINAL_SKLEARN_DIR = os.path.join(FINAL_ROOT_DIR, "sklearn")
FINAL_BUNDLE_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_model_bundle.pkl")
FINAL_MODEL_PATH = os.path.join(FINAL_SKLEARN_DIR, "fingerprint_model.pkl")
FINAL_SUMMARY_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_tuning_summary.csv")
FINAL_VALIDATION_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_validation_results.xlsx")
FINAL_IMPORTANCE_CSV_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_feature_importance.csv")
FINAL_IMPORTANCE_PLOT_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_feature_importance.png")
FINAL_REPORT_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_final_report.txt")
FINAL_VALIDATION_PLOT_PATH = os.path.join(FINAL_SKLEARN_DIR, "sklearn_validation_plots.png")
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER = 50
CV_FOLDS = 5
TOP_IMPORTANCE_N = 15


def ensure_results_dir() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs(FINAL_SKLEARN_DIR, exist_ok=True)





def _final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def compute_feature_importance(model, X, y, feature_cols):
    estimator = _final_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        method = "feature_importances_"
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        method = "abs(coef_)"
        importances = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        method = "permutation_importance"
        perm = permutation_importance(
            model,
            X,
            y,
            scoring="r2",
            n_repeats=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importances = np.asarray(perm.importances_mean, dtype=float)

    if importances.shape[0] != len(feature_cols):
        raise ValueError(
            f"重要性维度不匹配: importances={importances.shape[0]}, features={len(feature_cols)}"
        )

    importance_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    return method, importance_df


def save_feature_importance_plot(importance_df, method):
    plot_df = (
        importance_df.head(TOP_IMPORTANCE_N)
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_top_barh(
        ax,
        plot_df["Feature"],
        plot_df["Importance"],
        title=f"Sklearn Feature Importance (Top {len(plot_df)})",
        xlabel=f"Importance ({method})",
        base_color=COLORS["highlight"],
    )
    plt.tight_layout()
    plt.savefig(FINAL_IMPORTANCE_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def save_validation_plots(y_actual, y_predicted, results_df, model_name):
    """Generate 2x2 validation plots."""
    residuals = y_actual - y_predicted
    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Sklearn Validation Dashboard\nBest by CV: {model_name}", fontsize=16, fontweight="bold")

    # (1) Actual vs Predicted
    ax = axes[0, 0]
    hb = ax.hexbin(y_actual, y_predicted, gridsize=30, cmap=HEXBIN_CMAP, mincnt=1, linewidths=0)
    vmin = min(y_actual.min(), y_predicted.min())
    vmax = max(y_actual.max(), y_predicted.max())
    margin_ax = (vmax - vmin) * 0.05
    ax.plot(
        [vmin - margin_ax, vmax + margin_ax],
        [vmin - margin_ax, vmax + margin_ax],
        linestyle="--",
        linewidth=1.8,
        color=COLORS["accent"],
        label="Ideal fit",
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax)
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Count")
    add_stat_box(
        ax,
        f"R2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
        loc="lower right",
        facecolor=COLORS["neutral_light"],
    )

    # (2) Residual Distribution
    ax = axes[0, 1]
    ax.hist(residuals, bins=40, color=COLORS["secondary"], edgecolor="white", alpha=0.9)
    ax.axvline(x=0, color=COLORS["accent"], linestyle="--", linewidth=1.8)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    style_axis(ax, grid_axis="y")
    add_stat_box(
        ax,
        f"Mean = {residuals.mean():.4f}\nStd = {residuals.std():.4f}",
        loc="upper right",
        facecolor=COLORS["secondary_light"],
    )

    # (3) Residual vs Predicted — stat box shows within-RMSE coverage
    ax = axes[1, 0]
    ax.scatter(y_predicted, residuals, alpha=0.45, s=18, color=COLORS["accent"], edgecolors="none")
    ax.axhline(y=0, color=COLORS["accent"], linestyle="--", linewidth=1.8)
    ax.axhline(y=rmse, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax.axhline(y=-rmse, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Predicted")
    style_axis(ax)
    within_1rmse = np.mean(np.abs(residuals) <= rmse) * 100
    within_2rmse = np.mean(np.abs(residuals) <= 2 * rmse) * 100
    add_stat_box(
        ax,
        f"Within \u00b1RMSE: {within_1rmse:.1f}%\nWithin \u00b12\u00d7RMSE: {within_2rmse:.1f}%",
        loc="upper right",
        facecolor=COLORS["accent_light"],
        fontsize=9,
    )

    # (4) Model Comparison
    ax = axes[1, 1]
    plot_model_comparison(
        ax,
        results_df["Model"].tolist(),
        results_df["CV Val R2"].tolist(),
        results_df["Test R2"].tolist(),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FINAL_VALIDATION_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"可视化图表已保存: {FINAL_VALIDATION_PLOT_PATH}")


def build_model_search_space():
    return {
        "MLPRegressor": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            max_iter=5000,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__hidden_layer_sizes": [
                    (64,),
                    (128,),
                    (256,),
                    (64, 32),
                    (128, 64),
                    (256, 128),
                    (128, 64, 32),
                    (256, 128, 64),
                    (128, 64, 32, 16),
                ],
                "model__activation": ["relu", "tanh"],
                "model__solver": ["adam"],
                "model__alpha": [1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.05],
                "model__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
                "model__batch_size": [16, 32, 64],
            },
        },
        "XGBRegressor": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        XGBRegressor(
                            n_jobs=-1,
                            objective="reg:squarederror",
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "params": {
                "model__n_estimators": [100, 200, 300, 500, 800],
                "model__max_depth": [3, 4, 5, 6, 7, 9],
                "model__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
                "model__reg_alpha": [0, 0.01, 0.1, 0.5, 1, 5],
                "model__reg_lambda": [0.1, 0.5, 1, 5, 10],
                "model__min_child_weight": [1, 3, 5, 7],
                "model__gamma": [0, 0.01, 0.1, 0.5, 1],
            },
        },
        "RandomForestRegressor": {
            "pipeline": Pipeline(
                [("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE))]
            ),
            "params": {
                "model__n_estimators": [200, 300, 500, 800, 1000],
                "model__max_depth": [None, 5, 10, 15, 20, 30],
                "model__min_samples_split": [2, 3, 5, 8, 10],
                "model__min_samples_leaf": [1, 2, 3, 4],
                "model__max_features": ["sqrt", "log2", None, 0.5, 0.7],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline(
                [("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
            ),
            "params": {
                "model__n_estimators": [100, 200, 300, 500, 800],
                "model__max_depth": [2, 3, 4, 5, 6, 8],
                "model__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__min_samples_leaf": [1, 2, 3, 5, 8],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
                "model__loss": ["squared_error", "huber"],
            },
        },
    }


def main():
    ensure_results_dir()

    print("正在加载数据...")
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)
    feature_cols = [c for c in SELECTED_FEATURE_COLS if c in data.columns]
    missing_cols = [c for c in SELECTED_FEATURE_COLS if c not in data.columns]
    if missing_cols:
        print(f"警告：以下特征缺失，已自动跳过: {missing_cols}")
    if len(feature_cols) < 5:
        raise ValueError(f"有效特征过少: {len(feature_cols)}，请先检查 feature_config.py")

    X = data[feature_cols].values
    y = data[target_col].values

    split_indices = load_saved_split_indices(len(data), SPLIT_INDEX_PATH)
    if split_indices is not None:
        train_idx, test_idx = split_indices
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print("检测到特征筛选阶段切分索引，已复用相同 train/test 划分。")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print("未检测到可复用切分索引，使用当前脚本默认随机划分。")
    print(f"特征数: {len(feature_cols)}")
    print(f"数据准备完成：训练集 {X_train.shape[0]} 样本，测试集 {X_test.shape[0]} 样本")

    models_param_grid = build_model_search_space()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_overall = {
        "name": "",
        "model": None,
        "cv_val_r2": -float("inf"),
        "best_params": None,
    }

    print(f"\n开始自动寻优 (每个模型 {N_ITER} 组参数, {CV_FOLDS} 折交叉验证)...")
    for name, config in models_param_grid.items():
        print(f"\n正在调优: {name} ...")
        search = RandomizedSearchCV(
            config["pipeline"],
            config["params"],
            n_iter=N_ITER,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_train_r2 = float(search.cv_results_["mean_train_score"][search.best_index_])
        cv_val_r2 = float(search.best_score_)

        print(f"  最优参数: {search.best_params_}")
        print(f"  CV 训练 R2: {cv_train_r2:.4f}, CV 验证 R2: {cv_val_r2:.4f}")
        print(f"  测试集 R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results.append(
            {
                "Model": name,
                "Best Params": str(search.best_params_),
                "CV Train R2": cv_train_r2,
                "CV Val R2": cv_val_r2,
                "Test R2": r2,
                "Test MAE": mae,
                "Test RMSE": rmse,
            }
        )

        if cv_val_r2 > best_overall["cv_val_r2"]:
            best_overall["name"] = name
            best_overall["model"] = best_model
            best_overall["cv_val_r2"] = cv_val_r2
            best_overall["best_params"] = search.best_params_

    results_df = pd.DataFrame(results).sort_values(by="CV Val R2", ascending=False)
    print("\n" + "=" * 60)
    print("所有模型调优结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))

    if best_overall["model"] is None:
        raise RuntimeError("未得到可用模型，请检查搜索空间与数据。")

    bundle = {
        "model": best_overall["model"],
        "model_name": best_overall["name"],
        "feature_cols": feature_cols,
        "target_col": target_col,
        "best_params": best_overall["best_params"],
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
        "random_state": RANDOM_STATE,
    }

    with open(MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    with open(LEGACY_MODEL_PATH, "wb") as f:
        pickle.dump(best_overall["model"], f)
    results_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    # ========== 最终输出 (final_results/sklearn) ==========
    # 仅用测试集评估，防止数据泄漏
    y_test_pred = best_overall["model"].predict(X_test)
    final_r2 = r2_score(y_test, y_test_pred)
    final_mae = mean_absolute_error(y_test, y_test_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    validation_df = pd.DataFrame({
        "Sample": range(1, len(y_test) + 1),
        f"Actual ({target_col})": y_test,
        "Predicted": y_test_pred,
        "Residual": y_test - y_test_pred,
        "Abs Error": np.abs(y_test - y_test_pred),
    })

    importance_method, importance_df = compute_feature_importance(
        best_overall["model"], X_test, y_test, feature_cols
    )
    save_feature_importance_plot(importance_df, importance_method)
    save_validation_plots(y_test, y_test_pred, results_df, best_overall["name"])

    with open(FINAL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    with open(FINAL_MODEL_PATH, "wb") as f:
        pickle.dump(best_overall["model"], f)
    results_df.to_csv(FINAL_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    validation_df.to_excel(FINAL_VALIDATION_PATH, index=False)
    importance_df.to_csv(FINAL_IMPORTANCE_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Sklearn 最终结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"最佳模型: {best_overall['name']}\n")
        f.write(f"最佳参数: {best_overall['best_params']}\n")
        f.write(f"CV 最优 R2: {best_overall['cv_val_r2']:.4f}\n")
        f.write(
            f"测试集验证 R2/MAE/RMSE: {final_r2:.4f} / {final_mae:.4f} / {final_rmse:.4f}\n"
        )
        f.write(f"特征贡献方法: {importance_method}\n")
        f.write("\n模型搜索汇总见: sklearn_tuning_summary.csv\n")
        f.write("验证结果见: sklearn_validation_results.xlsx\n")
        f.write("特征贡献见: sklearn_feature_importance.csv / sklearn_feature_importance.png\n")

    print(f"\n最佳模型 (按 CV R2 选择): {best_overall['name']}")
    print(f"最佳 CV R2: {best_overall['cv_val_r2']:.4f}")
    print(f"统一模型包已保存: {MODEL_BUNDLE_PATH}")
    print(f"兼容模型已保存: {LEGACY_MODEL_PATH}")
    print(f"搜索摘要已保存: {SUMMARY_PATH}")
    print(f"\n最终输出目录: {FINAL_SKLEARN_DIR}")
    print(f"最终报告: {FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
