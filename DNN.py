import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from feature_config import ALL_FEATURE_COLS, SELECTED_FEATURE_COLS, resolve_target_col


# ========== 配置 ==========
DATA_PATH = "data/molecular_features.xlsx"
MODEL_PATH = "results/DNN.h5"
PREPROCESS_PATH = "results/DNN_preprocess.pkl"
LOSS_PLOT_PATH = "results/DNN_loss.png"
RUN_SUMMARY_PATH = "results/DNN_run_summary.csv"

# 可选: "selected" (16特征) / "all" (20特征)
FEATURE_MODE = "all"

# 训练设置
SEEDS = [42, 52, 62, 72, 82]
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.25  # train/val/test = 60/20/20


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim: int, seed: int) -> keras.Model:
    set_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="huber", metrics=["mae"])
    return model


def choose_features(mode: str):
    if mode == "all":
        return ALL_FEATURE_COLS
    if mode == "selected":
        return SELECTED_FEATURE_COLS
    raise ValueError("FEATURE_MODE 仅支持 'all' 或 'selected'")


def main():
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)
    feature_cols = choose_features(FEATURE_MODE)

    X = data[feature_cols].values
    y = data[target_col].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_IN_TRAINVAL, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    print(f"特征模式: {FEATURE_MODE}, 特征数: {len(feature_cols)}")
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
            X_train_scaled, y_train_scaled,
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
        all_results.append(
            {"seed": seed, "val_loss": val_loss, "R2": r2, "MAE": mae, "RMSE": rmse}
        )
        print(f"val_loss={val_loss:.4f}, test R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_history = history
            best_seed = seed

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
                "feature_mode": FEATURE_MODE,
                "seeds": SEEDS,
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
    plt.show()


if __name__ == "__main__":
    main()
