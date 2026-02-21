"""
DNN 模型验证脚本（测试集验证版）
优先使用训练阶段保存的预处理器，确保特征与标准化完全一致。
仅在测试集（20%）上评估，与 DNN_AutoTune / DNN特征贡献分析 保持一致。
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col
from utils.data_utils import load_saved_split_indices

# ========== 配置 ==========
MODEL_CANDIDATES = [
    "final_results/dnn/best_model.keras",
    "results/best_model.keras",
    "results/dnn_model.keras",
    "results/DNN.h5",
]
PREPROCESS_CANDIDATES = [
    "final_results/dnn/best_model_preprocess.pkl",
    "results/best_model_preprocess.pkl",
    "results/dnn_preprocess.pkl",
    "results/DNN_preprocess.pkl",
]
DATA_PATH = "data/molecular_features.xlsx"
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"

OUTPUT_DIR = "final_results/dnn"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dnn_validation_results.xlsx")

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = next((p for p in MODEL_CANDIDATES if Path(p).exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "未找到 DNN 模型文件。请先运行 DNN_AutoTune.py 生成 best_model.keras。"
        )

    preprocess_path = next((p for p in PREPROCESS_CANDIDATES if Path(p).exists()), None)
    if preprocess_path is None:
        raise FileNotFoundError(
            "未找到预处理器文件。\n"
            f"尝试过的路径: {PREPROCESS_CANDIDATES}\n"
            "请先运行 DNN_AutoTune.py 训练模型并保存预处理器。"
        )

    model = keras.models.load_model(model_path, compile=False)
    print(f"已加载模型: {model_path}")

    data = pd.read_excel(DATA_PATH)

    with Path(preprocess_path).open("rb") as f:
        meta = pickle.load(f)
    feature_cols = [c for c in meta["feature_cols"] if c in data.columns]
    target_col = meta.get("target_col", resolve_target_col(data.columns))
    if len(feature_cols) < 5:
        raise ValueError(f"预处理器中的有效特征不足: {len(feature_cols)}")
    scaler_X = meta["scaler_X"]
    scaler_y = meta.get("scaler_y")
    print(f"已加载预处理器: {preprocess_path} (features={len(feature_cols)})")

    # ========== 复用统一 train/test 划分，仅在测试集上评估 ==========
    split_result = load_saved_split_indices(len(data), SPLIT_INDEX_PATH)
    if split_result is not None:
        train_idx, test_idx = split_result
        split_source = f"已复用 split 索引: {SPLIT_INDEX_PATH}"
    else:
        all_idx = np.arange(len(data))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        split_source = f"未找到 split 索引，使用随机划分 (test_size={TEST_SIZE})"

    print(split_source)
    print(f"测试集样本数: {len(test_idx)} / 总样本数: {len(data)}")

    x_all_scaled = scaler_X.transform(data[feature_cols].values)
    x_test_scaled = x_all_scaled[test_idx]
    y_test = data[target_col].values[test_idx]

    y_pred_scaled = model.predict(x_test_scaled, verbose=0).reshape(-1, 1)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    else:
        y_pred = y_pred_scaled.ravel()

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n============ 模型验证结果（测试集）============")
    print(f"模型文件: {model_path}")
    print(f"测试集样本数: {len(y_test)}")
    print(f"R²  = {r2:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE= {rmse:.4f}")

    result_df = pd.DataFrame({
        "index": test_idx,
        f"Actual ({target_col})": y_test,
        "Predicted": y_pred,
        "Residual": y_test - y_pred,
        "Abs Error": np.abs(y_test - y_pred),
    }).sort_values("index").reset_index(drop=True)

    result_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n验证结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
