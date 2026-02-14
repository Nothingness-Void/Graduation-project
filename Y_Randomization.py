# -*- coding: utf-8 -*-
"""
Y_Randomization.py - Y-Randomization (Y-Scrambling) 验证

验证 QSAR 模型是否真正学到了特征与目标值之间的关系。

原理:
  1. 保持 X 不变，随机打乱 y_train
  2. 用打乱后的数据重新训练模型
  3. 重复 N 次，对比随机模型与真实模型的 R²
  4. 如果真实 R² 远高于随机 R² 分布 → 模型有效

用法:
    python Y_Randomization.py

输出:
    - final_results/sklearn/y_randomization.png    (分布图)
    - final_results/sklearn/y_randomization.csv    (详细数据)
"""
import os
import pickle
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from tqdm.auto import tqdm

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col

warnings.filterwarnings("ignore")

# ========== 配置 ==========
DATA_PATH = "data/molecular_features.xlsx"
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"
MODEL_BUNDLE_PATH = "results/sklearn_model_bundle.pkl"

OUTPUT_DIR = "final_results/sklearn"
PLOT_PATH = os.path.join(OUTPUT_DIR, "y_randomization.png")
CSV_PATH = os.path.join(OUTPUT_DIR, "y_randomization.csv")

N_ITERATIONS = 100       # 随机化轮数
CV_FOLDS = 5             # 交叉验证折数
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 中文 + 英文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_saved_split_indices(n_samples: int):
    """Load split indices if available and valid."""
    if not os.path.exists(SPLIT_INDEX_PATH):
        return None
    try:
        with np.load(SPLIT_INDEX_PATH, allow_pickle=False) as d:
            train_idx = d["train_idx"].astype(int)
            test_idx = d["test_idx"].astype(int)
            saved_n = int(d["n_samples"][0]) if "n_samples" in d else None
    except Exception:
        return None
    if saved_n is not None and saved_n != n_samples:
        return None
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    if np.intersect1d(train_idx, test_idx).size > 0:
        return None
    return train_idx, test_idx


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 加载模型 bundle
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(
            f"未找到模型文件: {MODEL_BUNDLE_PATH}\n"
            "请先运行 Sklearn_AutoTune.py 或 Sklearn.py 训练模型。"
        )
    with open(MODEL_BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    model_name = bundle.get("model_name", type(model).__name__)
    feature_cols = bundle.get("feature_cols", SELECTED_FEATURE_COLS)
    print("=" * 72)
    print("Y-Randomization (Y-Scrambling) 验证")
    print("=" * 72)
    print(f"[Step 1/6] 已加载最佳模型: {model_name}")
    print(f"          特征数: {len(feature_cols)}")
    print(f"          迭代轮数: {N_ITERATIONS}, CV折数: {CV_FOLDS}")

    # 2) 加载数据
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)
    missing_cols = [c for c in feature_cols if c not in data.columns]
    if missing_cols:
        raise KeyError(f"模型所需特征缺失 {len(missing_cols)} 个，例如: {missing_cols[:5]}")
    X = data[feature_cols].values
    y = data[target_col].values
    print(f"[Step 2/6] 数据加载完成: samples={len(y)}, target_col={target_col}")

    # 3) 复用 split
    split_result = load_saved_split_indices(len(data))
    if split_result is not None:
        train_idx, test_idx = split_result
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"[Step 3/6] 已复用 split 索引: {SPLIT_INDEX_PATH}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"[Step 3/6] 未找到 split 索引，已使用默认随机划分 (test_size={TEST_SIZE})。")

    print(f"          样本划分: train={len(y_train)}, test={len(y_test)}")

    # 4) 真实模型性能 (基线)
    t0 = time.time()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    real_model = clone(model)
    real_cv_scores = cross_val_score(real_model, X_train, y_train, cv=cv, scoring="r2")
    real_cv_r2 = float(np.mean(real_cv_scores))
    real_cv_std = float(np.std(real_cv_scores))

    real_model.fit(X_train, y_train)
    real_test_r2 = float(r2_score(y_test, real_model.predict(X_test)))

    print(f"[Step 4/6] 真实模型基线:")
    print(f"          CV R2 = {real_cv_r2:.4f} ± {real_cv_std:.4f}")
    print(f"          Test R2 = {real_test_r2:.4f}")

    # 5) Y-Randomization: N 轮
    print(f"[Step 5/6] 开始 Y-Randomization: {N_ITERATIONS} 轮")
    rng = np.random.RandomState(RANDOM_STATE)
    rand_results = []
    failed_count = 0

    for i in tqdm(range(N_ITERATIONS), desc="Y-Randomization", unit="iter", dynamic_ncols=True):
        y_shuffled = y_train.copy()
        rng.shuffle(y_shuffled)

        rand_model = clone(model)
        try:
            cv_scores = cross_val_score(rand_model, X_train, y_shuffled, cv=cv, scoring="r2")
            cv_r2 = float(np.mean(cv_scores))

            rand_model.fit(X_train, y_shuffled)
            test_r2 = float(r2_score(y_test, rand_model.predict(X_test)))
        except Exception:
            cv_r2 = -1.0
            test_r2 = -1.0
            failed_count += 1

        rand_results.append({"iteration": i + 1, "cv_r2": cv_r2, "test_r2": test_r2})

    # 6) 统计
    df = pd.DataFrame(rand_results)
    rand_cv_r2 = df["cv_r2"].values
    rand_test_r2 = df["test_r2"].values

    ge_cv = int(np.sum(rand_cv_r2 >= real_cv_r2))
    ge_test = int(np.sum(rand_test_r2 >= real_test_r2))
    # Permutation-style p-value correction to avoid optimistic zero.
    p_value_cv = float((ge_cv + 1) / (N_ITERATIONS + 1))
    p_value_test = float((ge_test + 1) / (N_ITERATIONS + 1))

    elapsed = time.time() - t0
    print(f"[Step 6/6] 统计结果")
    print("-" * 72)
    print(f"真实模型:     CV R2={real_cv_r2:.4f} ± {real_cv_std:.4f}, Test R2={real_test_r2:.4f}")
    print(
        f"随机模型(CV): mean={rand_cv_r2.mean():.4f}, std={rand_cv_r2.std():.4f}, "
        f"p95={np.quantile(rand_cv_r2, 0.95):.4f}, max={rand_cv_r2.max():.4f}"
    )
    print(
        f"随机模型(Test): mean={rand_test_r2.mean():.4f}, std={rand_test_r2.std():.4f}, "
        f"p95={np.quantile(rand_test_r2, 0.95):.4f}, max={rand_test_r2.max():.4f}"
    )
    print(f"p-value: CV={p_value_cv:.4f} ({ge_cv}/{N_ITERATIONS}), Test={p_value_test:.4f} ({ge_test}/{N_ITERATIONS})")
    print(f"异常轮数: {failed_count}")
    print(f"总耗时: {elapsed:.1f}s")
    print("-" * 72)

    if p_value_cv < 0.05 and p_value_test < 0.05:
        conclusion = "PASS - 模型有效 (p < 0.05)"
    else:
        conclusion = "FAIL - 模型可能存在过拟合或偶然相关"
    print(f"结论: {conclusion}")

    # 7) 保存 CSV
    df["real_cv_r2"] = real_cv_r2
    df["real_test_r2"] = real_test_r2
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"详细数据已保存: {CSV_PATH}")

    # 8) 可视化 (英文标注)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CV R2 distribution
    ax = axes[0]
    ax.hist(rand_cv_r2, bins=25, color="#6baed6", edgecolor="white", alpha=0.85, label="Randomized")
    ax.axvline(real_cv_r2, color="#e74c3c", linewidth=2.5, linestyle="--", label=f"Real model (R2={real_cv_r2:.4f})")
    ax.set_xlabel("CV R2", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Y-Randomization: CV R2  (p={p_value_cv:.4f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Test R2 distribution
    ax = axes[1]
    ax.hist(rand_test_r2, bins=25, color="#74c476", edgecolor="white", alpha=0.85, label="Randomized")
    ax.axvline(real_test_r2, color="#e74c3c", linewidth=2.5, linestyle="--", label=f"Real model (R2={real_test_r2:.4f})")
    ax.set_xlabel("Test R2", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Y-Randomization: Test R2  (p={p_value_test:.4f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    fig.suptitle(f"Y-Randomization Test — {model_name} ({N_ITERATIONS} iterations)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"分布图已保存: {PLOT_PATH}")


if __name__ == "__main__":
    main()
