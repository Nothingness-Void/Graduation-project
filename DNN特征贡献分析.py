"""
DNN 综合验证与特征贡献分析（严格对齐最新 AutoTune 产物）

输入:
- final_results/dnn/best_model.keras 或 results/best_model.keras
- final_results/dnn/best_model_preprocess.pkl 或 results/best_model_preprocess.pkl

输出:
- final_results/dnn/dnn_validation_plots.png
- final_results/dnn/dnn_validation_results.csv
- final_results/dnn/dnn_feature_importance.csv
"""

import os
import pickle
import tempfile
import warnings
import zipfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import keras
from keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_config import resolve_target_col
from utils.data_utils import load_saved_split_indices


warnings.filterwarnings("ignore")


# ========== 路径配置（仅最新） ==========
MODEL_CANDIDATES = [
    "final_results/dnn/best_model.keras",
    "results/best_model.keras",
]
PREPROCESS_CANDIDATES = [
    "final_results/dnn/best_model_preprocess.pkl",
    "results/best_model_preprocess.pkl",
]
DATA_PATH = "data/molecular_features.xlsx"
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"

FINAL_DNN_DIR = "final_results/dnn"
VALIDATION_PLOT_PATH = os.path.join(FINAL_DNN_DIR, "dnn_validation_plots.png")
VALIDATION_CSV_PATH = os.path.join(FINAL_DNN_DIR, "dnn_validation_results.csv")
IMPORTANCE_CSV_PATH = os.path.join(FINAL_DNN_DIR, "dnn_feature_importance.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_BACKGROUND = 256
MAX_EXPLAIN = 1024
PERM_REPEATS = 5


def ensure_dirs():
    os.makedirs(FINAL_DNN_DIR, exist_ok=True)


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def load_latest_preprocess():
    path = first_existing(PREPROCESS_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "未找到最新预处理器文件。请先运行 DNN_AutoTune.py 生成 best_model_preprocess.pkl。"
        )
    with open(path, "rb") as f:
        meta = pickle.load(f)

    required = {"feature_cols", "best_hp", "scaler_X"}
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"预处理器缺少关键字段: {missing}")
    return path, meta


def build_autotune_model(best_hp: dict, input_dim: int):
    model = keras.Sequential(name="dnn_autotune_rebuilt")
    model.add(keras.layers.Input(shape=(input_dim,)))

    n_layers = int(best_hp.get("n_layers", 2))
    use_bn = bool(best_hp.get("use_batchnorm", False))
    l2_val = float(best_hp.get("l2_reg", 0.0) or 0.0)

    for i in range(n_layers):
        units = int(best_hp.get(f"units_layer_{i}", 32))
        dropout = float(best_hp.get(f"dropout_layer_{i}", 0.0) or 0.0)
        model.add(
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_val) if l2_val > 0 else None,
            )
        )
        if use_bn:
            model.add(keras.layers.BatchNormalization())
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1, activation="linear"))

    lr = float(best_hp.get("learning_rate", 1e-3))
    loss_name = best_hp.get("loss", "huber")
    if loss_name == "huber":
        loss = keras.losses.Huber(delta=1.0)
    elif loss_name == "mae":
        loss = "mae"
    else:
        loss = "mse"

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=["mae"])
    return model


def _manual_load_weights_from_keras_archive(model, keras_path: str):
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(keras_path, "r") as zf:
            if "model.weights.h5" not in zf.namelist():
                raise ValueError(f"{keras_path} 中缺少 model.weights.h5")
            zf.extract("model.weights.h5", td)
        weights_path = os.path.join(td, "model.weights.h5")

        with h5py.File(weights_path, "r") as h5f:
            keys = list(h5f.keys())

            def find_group_for_layer(layer_name: str):
                lname = layer_name.lower()
                for k in keys:
                    if k.lower().endswith(lname):
                        return k
                return None

            loaded_layers = 0
            for layer in model.layers:
                expected = len(layer.weights)
                if expected == 0:
                    continue

                gkey = find_group_for_layer(layer.name)
                if gkey is None:
                    continue
                group = h5f[gkey]
                if "vars" not in group:
                    continue

                vars_group = group["vars"]
                arrays = []
                ok = True
                for i in range(expected):
                    if str(i) not in vars_group:
                        ok = False
                        break
                    arrays.append(vars_group[str(i)][()])
                if not ok:
                    continue

                try:
                    layer.set_weights(arrays)
                    loaded_layers += 1
                except Exception:
                    continue

            if loaded_layers == 0:
                raise ValueError("手动权重装载失败：没有任何层成功装载。")


def load_latest_model(meta):
    model_path = first_existing(MODEL_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError("未找到最新模型文件 best_model.keras。请先运行 DNN_AutoTune.py。")

    # 1) 直接加载
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model, model_path, "direct_load"
    except Exception as direct_err:
        # 2) 回退：按 best_hp 重建后手动装载权重
        input_dim = len(meta["feature_cols"])
        model = build_autotune_model(meta["best_hp"], input_dim)
        try:
            model.load_weights(model_path)
            return model, model_path, "rebuild+load_weights"
        except Exception:
            _manual_load_weights_from_keras_archive(model, model_path)
            return model, model_path, f"rebuild+manual_weights (direct_err={type(direct_err).__name__})"


def get_train_test_indices(n_samples: int):
    split_indices = load_saved_split_indices(n_samples, SPLIT_INDEX_PATH)
    if split_indices is not None:
        return split_indices, "saved_split"
    all_idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return (train_idx, test_idx), "random_split"


def predict_with_optional_inverse(model, x_scaled, scaler_y):
    y_pred_scaled = model.predict(x_scaled, verbose=0).reshape(-1, 1)
    if scaler_y is None:
        return y_pred_scaled.ravel()
    try:
        return scaler_y.inverse_transform(y_pred_scaled).ravel()
    except Exception:
        return y_pred_scaled.ravel()


def compute_permutation_like_importance(model, x_test_scaled, y_test, scaler_y):
    rng = np.random.RandomState(RANDOM_STATE)
    base_pred = predict_with_optional_inverse(model, x_test_scaled, scaler_y)
    base_r2 = r2_score(y_test, base_pred)

    importances = np.zeros(x_test_scaled.shape[1], dtype=float)
    for j in range(x_test_scaled.shape[1]):
        drops = []
        for _ in range(PERM_REPEATS):
            x_perm = x_test_scaled.copy()
            perm_idx = rng.permutation(x_perm.shape[0])
            x_perm[:, j] = x_perm[perm_idx, j]
            y_perm_pred = predict_with_optional_inverse(model, x_perm, scaler_y)
            drops.append(base_r2 - r2_score(y_test, y_perm_pred))
        importances[j] = float(np.mean(drops))
    return importances


def compute_importance(model, x_train_scaled, x_test_scaled, y_test, feature_cols, scaler_y):
    bg_n = min(MAX_BACKGROUND, len(x_train_scaled))
    ex_n = min(MAX_EXPLAIN, len(x_test_scaled))
    x_bg = x_train_scaled[:bg_n]
    x_explain = x_test_scaled[:ex_n]

    try:
        explainer = shap.GradientExplainer(model, x_bg)
        shap_values = explainer.shap_values(x_explain)
        shap_out = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_out = np.asarray(shap_out)
        if shap_out.ndim == 3 and shap_out.shape[-1] == 1:
            shap_out = shap_out[:, :, 0]
        if shap_out.ndim == 1:
            shap_out = shap_out.reshape(-1, 1)
        importance = np.mean(np.abs(shap_out), axis=0).astype(float).ravel()
        method = "SHAP Mean |Value|"
    except Exception:
        importance = compute_permutation_like_importance(model, x_test_scaled, y_test, scaler_y)
        method = "Permutation-like Delta R2"

    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return method, imp_df


def save_dashboard_plot(y_test, y_pred, imp_df, method, model_label):
    residuals = y_test - y_pred
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"DNN Validation Dashboard — {model_label}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(y_test, y_pred, alpha=0.45, s=16, c="#2196F3", edgecolors="none")
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05 if vmax > vmin else 0.1
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], "r--", linewidth=1.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.text(
        0.95,
        0.05,
        f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85),
    )

    ax = axes[0, 1]
    ax.hist(residuals, bins=35, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.text(
        0.95,
        0.95,
        f"Mean = {residuals.mean():.4f}\nStd = {residuals.std():.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.85),
    )

    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.45, s=16, c="#FF9800", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Predicted (Heteroscedasticity Check)")

    ax = axes[1, 1]
    top_df = imp_df.head(12).sort_values("Importance", ascending=True)
    ax.barh(top_df["Feature"], top_df["Importance"], color="#7E57C2", alpha=0.9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance ({method})")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(VALIDATION_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    preprocess_path, meta = load_latest_preprocess()
    model, model_path, load_mode = load_latest_model(meta)

    data = pd.read_excel(DATA_PATH)
    target_col = meta.get("target_col", resolve_target_col(data.columns))
    feature_cols = list(meta["feature_cols"])

    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(f"最新预处理器要求的特征缺失: {missing}")

    x_raw = data[feature_cols].values
    y_all = data[target_col].values
    scaler_x = meta.get("scaler_X")
    scaler_y = meta.get("scaler_y")
    if scaler_x is None:
        raise ValueError("best_model_preprocess.pkl 中缺少 scaler_X，无法对齐最新训练流程。")

    (train_idx, test_idx), split_source = get_train_test_indices(len(data))
    x_scaled = scaler_x.transform(x_raw)
    x_train_scaled = x_scaled[train_idx]
    x_test_scaled = x_scaled[test_idx]
    y_test = y_all[test_idx]
    y_pred = predict_with_optional_inverse(model, x_test_scaled, scaler_y)

    method, imp_df = compute_importance(
        model=model,
        x_train_scaled=x_train_scaled,
        x_test_scaled=x_test_scaled,
        y_test=y_test,
        feature_cols=feature_cols,
        scaler_y=scaler_y,
    )

    val_df = pd.DataFrame(
        {
            "index": test_idx,
            f"Actual ({target_col})": y_test,
            "Predicted": y_pred,
            "Residual": y_test - y_pred,
        }
    ).sort_values("index")
    val_df.to_csv(VALIDATION_CSV_PATH, index=False, encoding="utf-8-sig")
    imp_df.to_csv(IMPORTANCE_CSV_PATH, index=False, encoding="utf-8-sig")

    model_label = f"{Path(model_path).name} | {load_mode}"
    save_dashboard_plot(y_test, y_pred, imp_df, method, model_label)

    print(f"模型来源: {model_path}")
    print(f"预处理来源: {preprocess_path}")
    print(f"加载模式: {load_mode}")
    print(f"数据切分来源: {split_source}")
    print(f"\n综合验证图已保存: {VALIDATION_PLOT_PATH}")
    print(f"验证明细已保存: {VALIDATION_CSV_PATH}")
    print(f"特征贡献已保存: {IMPORTANCE_CSV_PATH}")


if __name__ == "__main__":
    main()
