"""
DNN 特征贡献分析脚本（SHAP）
优先读取训练保存的预处理器，确保与训练输入一致。
"""

import pickle
from pathlib import Path
import shap
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_config import SELECTED_FEATURE_COLS


# ========== 配置 ==========
MODEL_CANDIDATES = ["results/dnn_model.keras", "results/DNN.h5", "results/best_model.keras"]
PREPROCESS_CANDIDATES = [
    "results/dnn_preprocess.pkl",
    "results/DNN_preprocess.pkl",
    "results/best_model_preprocess.pkl",
]
DATA_PATH = "data/molecular_features.xlsx"
OUTPUT_PATH = "results/dnn_shap_analysis.png"
MAX_BACKGROUND = 256
MAX_EXPLAIN = 1024
RANDOM_STATE = 42


def main():
    model_path = next((p for p in MODEL_CANDIDATES if Path(p).exists()), None)
    if model_path is None:
        raise FileNotFoundError("未找到 DNN 模型文件。请先运行 DNN.py（或 DNN_AutoTune.py）。")
    preprocess_path = next((p for p in PREPROCESS_CANDIDATES if Path(p).exists()), None)

    model = keras.models.load_model(model_path, compile=False)
    print(f"已加载模型: {model_path}")

    data = pd.read_excel(DATA_PATH)

    if preprocess_path is not None:
        with Path(preprocess_path).open("rb") as f:
            meta = pickle.load(f)
        feature_cols = meta.get("feature_cols", SELECTED_FEATURE_COLS)
        feature_cols = [c for c in feature_cols if c in data.columns]
        if len(feature_cols) < 5:
            raise ValueError(f"预处理器中的有效特征不足: {len(feature_cols)}")
        scaler_X = meta.get("scaler_X")
        if scaler_X is None:
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(data[feature_cols].values)
            print("预处理器中缺少 scaler_X，已回退到即时标准化。")
        else:
            X_scaled = scaler_X.transform(data[feature_cols].values)
        print(f"已加载预处理器: {preprocess_path} (features={len(feature_cols)})")
    else:
        feature_cols = SELECTED_FEATURE_COLS
        feature_cols = [c for c in feature_cols if c in data.columns]
        if len(feature_cols) < 5:
            raise ValueError(f"有效特征不足: {len(feature_cols)}，请检查 feature_config.py")
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(data[feature_cols].values)
        print("未找到预处理器，已使用回退模式（fit_transform 全量数据）。")

    X_val_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # SHAP 对样本数较敏感，限制背景集和解释集可显著降低耗时
    bg_n = min(MAX_BACKGROUND, len(X_val_df))
    ex_n = min(MAX_EXPLAIN, len(X_val_df))
    X_bg = X_val_df.sample(n=bg_n, random_state=RANDOM_STATE)
    X_explain = X_val_df.sample(n=ex_n, random_state=RANDOM_STATE + 1)

    print(f"正在计算 SHAP 值... background={bg_n}, explain={ex_n}")
    explainer = shap.GradientExplainer(model, X_bg.values)
    shap_values = explainer.shap_values(X_explain.values)
    shap_values_output = shap_values[0] if isinstance(shap_values, list) else shap_values

    mean_abs_shap = np.mean(np.abs(shap_values_output), axis=0)
    mean_shap = np.mean(shap_values_output, axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    feature_arr = np.array(feature_cols)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    axes[0].barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], align="center")
    axes[0].set_yticks(range(len(sorted_idx)))
    axes[0].set_yticklabels(feature_arr[sorted_idx])
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title("Feature Importance (Absolute)")

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in mean_shap[sorted_idx]]
    axes[1].barh(range(len(sorted_idx)), mean_shap[sorted_idx], align="center", color=colors)
    axes[1].set_yticks(range(len(sorted_idx)))
    axes[1].set_yticklabels(feature_arr[sorted_idx])
    axes[1].set_xlabel("Mean SHAP Value")
    axes[1].set_title("Feature Importance (Signed)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\n特征贡献分析图已保存至: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
