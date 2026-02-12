"""
DNN 模型验证脚本
优先使用训练阶段保存的预处理器，确保特征与标准化完全一致
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from feature_config import SELECTED_FEATURE_COLS, resolve_target_col

# ========== 配置 ==========
MODEL_CANDIDATES = ["results/dnn_model.keras", "results/DNN.h5", "results/best_model.keras"]
PREPROCESS_CANDIDATES = [
    "results/dnn_preprocess.pkl",
    "results/DNN_preprocess.pkl",
    "results/best_model_preprocess.pkl",
]
DATA_PATH = "data/molecular_features.xlsx"
OUTPUT_PATH = "results/dnn_validation_results.xlsx"


def main():
    model_path = next((p for p in MODEL_CANDIDATES if Path(p).exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "未找到 DNN 模型文件。请先运行 DNN.py（或 DNN_AutoTune.py）。"
        )

    preprocess_path = next((p for p in PREPROCESS_CANDIDATES if Path(p).exists()), None)

    model = keras.models.load_model(model_path, compile=False)
    print(f"已加载模型: {model_path}")

    data = pd.read_excel(DATA_PATH)

    if preprocess_path is not None:
        with Path(preprocess_path).open("rb") as f:
            meta = pickle.load(f)
        feature_cols = meta["feature_cols"]
        target_col = meta.get("target_col", resolve_target_col(data.columns))
        feature_cols = [c for c in feature_cols if c in data.columns]
        if len(feature_cols) < 5:
            raise ValueError(f"预处理器中的有效特征不足: {len(feature_cols)}")
        scaler_X = meta["scaler_X"]
        scaler_y = meta.get("scaler_y")
        print(f"已加载预处理器: {preprocess_path} (features={len(feature_cols)})")
        X_val_scaled = scaler_X.transform(data[feature_cols].values)
    else:
        # 兼容旧模型：若无预处理器文件，回退到当前统一配置
        feature_cols = SELECTED_FEATURE_COLS
        target_col = resolve_target_col(data.columns)
        feature_cols = [c for c in feature_cols if c in data.columns]
        if len(feature_cols) < 5:
            raise ValueError(f"有效特征不足: {len(feature_cols)}，请检查 feature_config.py")
        scaler_X = StandardScaler()
        scaler_y = None
        X_val_scaled = scaler_X.fit_transform(data[feature_cols].values)
        print("未找到预处理器文件，已使用回退模式（fit_transform 全量数据）")

    y_true = data[target_col].values
    y_pred_scaled = model.predict(X_val_scaled, verbose=0).reshape(-1, 1)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    else:
        y_pred = y_pred_scaled.ravel()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n============ 模型验证结果 ============")
    print(f"模型文件: {model_path}")
    print(f"验证样本数: {len(y_true)}")
    print(f"R2值为：{r2:.4f}")
    print(f"MAE(平均绝对误差)值为：{mae:.4f}")
    print(f"RMSE(均方根误差)值为：{rmse:.4f}")

    data["Predicted χ-result"] = y_pred
    data["Residual"] = y_true - y_pred
    data.to_excel(OUTPUT_PATH, index=False)
    print(f"\n验证结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
