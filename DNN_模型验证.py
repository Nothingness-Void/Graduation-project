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
MODEL_PATH = "results/DNN.h5"
PREPROCESS_PATH = "results/DNN_preprocess.pkl"
DATA_PATH = "data/molecular_features.xlsx"
OUTPUT_PATH = "results/DNN_validation_results.xlsx"


def main():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"已加载模型: {MODEL_PATH}")

    data = pd.read_excel(DATA_PATH)

    preprocess_file = Path(PREPROCESS_PATH)
    if preprocess_file.exists():
        with preprocess_file.open("rb") as f:
            meta = pickle.load(f)
        feature_cols = meta["feature_cols"]
        target_col = meta.get("target_col", resolve_target_col(data.columns))
        scaler_X = meta["scaler_X"]
        scaler_y = meta.get("scaler_y")
        print(f"已加载预处理器: {PREPROCESS_PATH} (features={len(feature_cols)})")
        X_val_scaled = scaler_X.transform(data[feature_cols].values)
    else:
        # 兼容旧模型：若无预处理器文件，回退到当前统一配置
        feature_cols = SELECTED_FEATURE_COLS
        target_col = resolve_target_col(data.columns)
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
    print(f"模型文件: {MODEL_PATH}")
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
