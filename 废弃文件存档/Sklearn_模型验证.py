"""
Sklearn 模型验证脚本
优先读取模型包（含特征配置），保证训练/验证使用同一套特征定义
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from feature_config import SELECTED_FEATURE_COLS, resolve_target_col

# ========== 配置 ==========
BUNDLE_PATH = "results/sklearn_model_bundle.pkl"
LEGACY_MODEL_PATHS = ["results/fingerprint_model.pkl", "results/best_model_sklearn.pkl"]
DATA_PATH = "data/molecular_features.xlsx"
OUTPUT_PATH = "results/sklearn_validation_results.xlsx"


def main():
    data = pd.read_excel(DATA_PATH)

    # 优先加载新模型包；如果不存在则回退到旧模型格式
    if Path(BUNDLE_PATH).exists():
        with open(BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        feature_cols = [c for c in bundle["feature_cols"] if c in data.columns]
        if len(feature_cols) < 5:
            raise ValueError(f"模型包中的有效特征不足: {len(feature_cols)}")
        target_col = bundle.get("target_col", resolve_target_col(data.columns))
        if target_col not in data.columns:
            target_col = resolve_target_col(data.columns)
        print(f"已加载模型包: {BUNDLE_PATH} ({model.__class__.__name__})")
    else:
        model_path = next((p for p in LEGACY_MODEL_PATHS if Path(p).exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"未找到模型文件。请先运行 Sklearn_AutoTune.py，期望: {BUNDLE_PATH}"
            )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feature_cols = [c for c in SELECTED_FEATURE_COLS if c in data.columns]
        if len(feature_cols) < 5:
            raise ValueError(f"有效特征不足: {len(feature_cols)}，请检查 feature_config.py")
        target_col = resolve_target_col(data.columns)
        print(f"已加载旧模型: {model_path} ({model.__class__.__name__})")
        print("提示: 未找到模型包，已回退到默认特征配置")

    X_val = data[feature_cols]
    y_val = data[target_col].values

    # 若模型是 Pipeline，会自动处理缩放；若不是，也直接按模型定义预测
    y_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print("\n============ 模型验证结果 ============")
    print(f"模型类型: {model.__class__.__name__}")
    print(f"验证样本数: {len(y_val)}")
    print(f"R2值为：{r2:.4f}")
    print(f"MAE(平均绝对误差)值为：{mae:.4f}")
    print(f"RMSE(均方根误差)值为：{rmse:.4f}")

    data["Predicted χ-result"] = y_pred
    data["Residual"] = y_val - y_pred
    data.to_excel(OUTPUT_PATH, index=False)
    print(f"\n验证结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
