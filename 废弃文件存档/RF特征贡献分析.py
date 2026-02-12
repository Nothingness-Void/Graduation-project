"""
Sklearn 特征贡献分析脚本
优先读取 sklearn_model_bundle.pkl；若无则回退到旧模型文件。
支持 feature_importances_ / coef_ / permutation_importance 三种方式。
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col


# ========== 配置 ==========
BUNDLE_PATH = "results/sklearn_model_bundle.pkl"
LEGACY_MODEL_PATHS = ["results/fingerprint_model.pkl", "results/best_model_sklearn.pkl"]
DATA_PATH = "data/molecular_features.xlsx"
OUTPUT_PATH = "results/sklearn_feature_importance.png"


def load_model_and_meta():
    if Path(BUNDLE_PATH).exists():
        with open(BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        return (
            bundle["model"],
            bundle.get("feature_cols", SELECTED_FEATURE_COLS),
            bundle.get("target_col"),
            BUNDLE_PATH,
        )

    legacy_path = next((p for p in LEGACY_MODEL_PATHS if Path(p).exists()), None)
    if legacy_path is None:
        raise FileNotFoundError(
            "未找到可用 Sklearn 模型。请先运行 Sklearn_AutoTune.py。"
        )
    with open(legacy_path, "rb") as f:
        model = pickle.load(f)
    return model, SELECTED_FEATURE_COLS, None, legacy_path


def _final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def main():
    model, feature_cols, target_col, model_path = load_model_and_meta()
    print(f"已加载模型: {model_path} ({model.__class__.__name__})")

    data = pd.read_excel(DATA_PATH)
    feature_cols = [c for c in feature_cols if c in data.columns]
    if not feature_cols:
        raise ValueError("模型特征在数据中均不存在，请检查 feature_config.py 或模型包。")

    X = data[feature_cols]
    if target_col is None or target_col not in data.columns:
        target_col = resolve_target_col(data.columns)
    y = data[target_col].values

    estimator = _final_estimator(model)
    method = ""

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        importances = np.abs(coef).ravel()
        method = "abs(coef_)"
    else:
        print("模型不支持内置重要性，正在使用 permutation_importance ...")
        perm = permutation_importance(model, X, y, scoring="r2", n_repeats=10, random_state=42, n_jobs=-1)
        importances = np.asarray(perm.importances_mean, dtype=float)
        method = "permutation_importance"

    if importances.shape[0] != len(feature_cols):
        raise ValueError(
            f"重要性维度不匹配: importances={importances.shape[0]}, features={len(feature_cols)}"
        )

    sorted_idx = np.argsort(importances)
    feature_arr = np.array(feature_cols)

    print("\n============ 特征重要性排序 ============")
    print(f"方法: {method}")
    for i in reversed(sorted_idx):
        print(f"  {feature_arr[i]:35s}  {importances[i]:.6f}")

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), feature_arr[sorted_idx])
    plt.xlabel(f"Importance ({method})")
    plt.title(f"Feature Importances ({model.__class__.__name__})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\n特征重要性图已保存至: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
