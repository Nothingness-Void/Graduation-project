import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col


# ========== 配置 ==========
DATA_PATH = "data/molecular_features.xlsx"
MODEL_PATH = "results/dnn_model.keras"
PREPROCESS_PATH = "results/dnn_preprocess.pkl"
LOSS_PLOT_PATH = "results/dnn_loss.png"
RUN_SUMMARY_PATH = "results/dnn_run_summary.csv"
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"

SEEDS = [42, 52, 62, 72, 82]
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.25  # train/val/test = 60/20/20
RANDOM_STATE = 42


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(input_dim: int, seed: int) -> keras.Model:
    set_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(48, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(12, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="huber", metrics=["mae"])
    return model


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
    os.makedirs("results", exist_ok=True)

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

    # 优先复用上游保存的 train/test split
    split_result = load_saved_split_indices(len(data))
    if split_result is not None:
        train_idx, test_idx = split_result
        X_trainval, X_test = X[train_idx], X[test_idx]
        y_trainval, y_test = y[train_idx], y[test_idx]
        print("检测到特征筛选阶段切分索引，已复用相同 train/test 划分。")
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print("未检测到可复用切分索引，使用当前脚本默认随机划分。")
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_IN_TRAINVAL, random_state=RANDOM_STATE
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    print(f"特征数: {len(feature_cols)}")
    print(f"样本划分: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    all_results = []
    best_model = None
    best_history = None
    best_seed = None
    best_val_loss = float("inf")

    for i, seed in enumerate(SEEDS, start=1):
        print(f"\n--- Run {i}/{len(SEEDS)} (seed={seed}) ---")
        model = build_model(X_train_scaled.shape[1], seed)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ]
        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=1000,
            batch_size=16,
            callbacks=callbacks,
            verbose=0,
        )

        y_pred_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        val_loss = float(np.min(history.history["val_loss"]))
        all_results.append({"seed": seed, "val_loss": val_loss, "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"val_loss={val_loss:.4f}, test R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_history = history
            best_seed = seed

    if best_model is None:
        raise RuntimeError("未得到可用 DNN 模型。")

    results_df = pd.DataFrame(all_results)
    print("\n==================== 多次运行统计 ====================")
    print(results_df.to_string(index=False))
    print(
        f"\nR2 mean={results_df['R2'].mean():.4f}, std={results_df['R2'].std():.4f}, "
        f"best={results_df['R2'].max():.4f}"
    )
    print(f"按 val_loss 选中的最佳 seed: {best_seed}, best val_loss={best_val_loss:.4f}")

    best_model.save(MODEL_PATH)
    with open(PREPROCESS_PATH, "wb") as f:
        pickle.dump(
            {
                "feature_cols": feature_cols,
                "target_col": target_col,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "seeds": SEEDS,
                "best_seed": best_seed,
            },
            f,
        )
    results_df.to_csv(RUN_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"\n模型已保存: {MODEL_PATH}")
    print(f"预处理器已保存: {PREPROCESS_PATH}")
    print(f"运行汇总已保存: {RUN_SUMMARY_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(best_history.history["loss"], label="train")
    plt.plot(best_history.history["val_loss"], label="val")
    plt.title(f"DNN Loss (best seed={best_seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=160)
    plt.close()
    print(f"Loss 曲线已保存: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
