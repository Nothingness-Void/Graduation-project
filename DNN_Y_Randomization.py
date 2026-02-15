# -*- coding: utf-8 -*-
"""
DNN_Y_Randomization.py

Y-randomization (Y-scrambling) validation for the DNN pipeline.

Workflow:
1) Reuse the saved train/test split indices to keep evaluation consistent.
2) Train a baseline DNN with real y and evaluate on test.
3) Repeat N times:
   - shuffle y_train and y_val independently
   - retrain the same DNN
   - evaluate on the same untouched y_test
4) Compare baseline test R2 with randomized distribution.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col
from utils.data_utils import load_saved_split_indices

warnings.filterwarnings("ignore")


# ========== Paths ==========
DATA_PATH = "data/molecular_features.xlsx"
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"
PREPROCESS_CANDIDATES = [
    "results/best_model_preprocess.pkl",
    "final_results/dnn/best_model_preprocess.pkl",
    "results/dnn_preprocess.pkl",
]

OUTPUT_DIR = "final_results/dnn"
CSV_PATH = os.path.join(OUTPUT_DIR, "dnn_y_randomization.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "dnn_y_randomization.png")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "dnn_y_randomization_summary.txt")


# ========== Config ==========
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.25
RANDOM_STATE = 42

N_ITERATIONS = 100
EPOCHS = 1000
SCALE_Y = True


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def load_autotune_artifacts():
    """Load saved preprocess/hyperparameter bundle from DNN_AutoTune if available."""
    for path in PREPROCESS_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            return bundle, path
        except Exception:
            continue
    return None, None


def build_model(input_dim: int, seed: int, hp_values=None) -> keras.Model:
    """Build DNN; if hp_values is provided, mirror DNN_AutoTune best architecture."""
    set_seed(seed)
    if hp_values:
        n_layers = int(hp_values.get("n_layers", 2))
        use_batchnorm = bool(hp_values.get("use_batchnorm", False))
        l2_reg = float(hp_values.get("l2_reg", 0.0) or 0.0)

        model = keras.Sequential([keras.layers.Input(shape=(input_dim,))])
        for i in range(n_layers):
            units = int(hp_values.get(f"units_layer_{i}", 32))
            dropout = float(hp_values.get(f"dropout_layer_{i}", 0.0) or 0.0)
            model.add(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
                )
            )
            if use_batchnorm:
                model.add(keras.layers.BatchNormalization())
            if dropout > 0:
                model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(1, activation="linear"))
        learning_rate = float(hp_values.get("learning_rate", 1e-3))
        loss_name = hp_values.get("loss", "huber")
        if loss_name == "huber":
            loss = keras.losses.Huber(delta=1.0)
        elif loss_name == "mae":
            loss = "mae"
        else:
            loss = "mse"
    else:
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
        learning_rate = 1e-3
        loss = "huber"

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=["mae"])
    return model





def train_and_eval_once(
    X_train_scaled,
    y_train_fit,
    X_val_scaled,
    y_val_fit,
    X_test_scaled,
    y_test,
    scaler_y,
    seed: int,
    hp_values=None,
    batch_size: int = 16,
):
    model = build_model(X_train_scaled.shape[1], seed=seed, hp_values=hp_values)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    ]
    history = model.fit(
        X_train_scaled,
        y_train_fit,
        validation_data=(X_val_scaled, y_val_fit),
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    else:
        y_pred = y_pred_scaled.ravel()
    test_r2 = float(r2_score(y_test, y_pred))
    best_val_loss = float(np.min(history.history["val_loss"]))
    return test_r2, best_val_loss, len(history.history["loss"])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("DNN Y-Randomization (Y-Scrambling) Validation")
    print("=" * 72)

    preprocess_bundle, preprocess_source = load_autotune_artifacts()

    # 1) Load data
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)

    if preprocess_bundle and preprocess_bundle.get("feature_cols"):
        requested_features = list(preprocess_bundle["feature_cols"])
    else:
        requested_features = list(SELECTED_FEATURE_COLS)

    feature_cols = [c for c in requested_features if c in data.columns]
    missing_cols = [c for c in requested_features if c not in data.columns]
    if missing_cols:
        print(f"Warning: missing features skipped: {missing_cols}")
    if len(feature_cols) < 5:
        raise ValueError(f"Too few valid features: {len(feature_cols)}")

    hp_values = preprocess_bundle.get("best_hp") if preprocess_bundle else None
    batch_size = 16
    if hp_values is not None:
        batch_size = int(hp_values.get("batch_size", 16))

    X = data[feature_cols].values
    y = data[target_col].values
    print(f"[Step 1/6] Data loaded: samples={len(y)}, features={len(feature_cols)}, target={target_col}")
    if preprocess_source:
        print(f"           model config source: {preprocess_source}")
        print(f"           aligned with AutoTune best HP, batch_size={batch_size}")
    else:
        print("           model config source: fallback fixed architecture")

    # 2) Split (reuse saved split if available)
    split_result = load_saved_split_indices(len(data), SPLIT_INDEX_PATH)
    if split_result is not None:
        train_idx, test_idx = split_result
        X_trainval, X_test = X[train_idx], X[test_idx]
        y_trainval, y_test = y[train_idx], y[test_idx]
        print(f"[Step 2/6] Reused split: {SPLIT_INDEX_PATH}")
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"[Step 2/6] Split file not found, fallback random split (test_size={TEST_SIZE})")

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_IN_TRAINVAL, random_state=RANDOM_STATE
    )
    print(f"           split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # 3) Preprocess (fit on train only)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = None
    y_train_fit = y_train.copy()
    y_val_fit = y_val.copy()
    if SCALE_Y:
        scaler_y = StandardScaler()
        y_train_fit = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_fit = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    print(f"[Step 3/6] Preprocess done (scale_y={SCALE_Y})")

    # 4) Baseline with real y
    t0 = time.time()
    real_test_r2, real_best_val_loss, real_epochs = train_and_eval_once(
        X_train_scaled,
        y_train_fit,
        X_val_scaled,
        y_val_fit,
        X_test_scaled,
        y_test,
        scaler_y,
        seed=RANDOM_STATE,
        hp_values=hp_values,
        batch_size=batch_size,
    )
    print("[Step 4/6] Baseline model")
    print(
        f"           test R2={real_test_r2:.4f}, best val_loss={real_best_val_loss:.4f}, "
        f"epochs={real_epochs}"
    )

    # 5) Y-randomization
    rng = np.random.RandomState(RANDOM_STATE)
    rand_rows = []
    print(f"[Step 5/6] Start Y-randomization: iterations={N_ITERATIONS}")
    for i in tqdm(range(N_ITERATIONS), desc="DNN Y-Randomization", unit="iter", dynamic_ncols=True):
        y_train_shuffled = y_train_fit.copy()
        y_val_shuffled = y_val_fit.copy()
        rng.shuffle(y_train_shuffled)
        rng.shuffle(y_val_shuffled)

        try:
            test_r2, best_val_loss, n_epochs = train_and_eval_once(
                X_train_scaled,
                y_train_shuffled,
                X_val_scaled,
                y_val_shuffled,
                X_test_scaled,
                y_test,
                scaler_y,
                seed=RANDOM_STATE + i + 1,
                hp_values=hp_values,
                batch_size=batch_size,
            )
            failed = 0
        except Exception:
            test_r2, best_val_loss, n_epochs = -1.0, 1e6, 0
            failed = 1

        rand_rows.append(
            {
                "iteration": i + 1,
                "test_r2": test_r2,
                "best_val_loss": best_val_loss,
                "epochs": n_epochs,
                "failed": failed,
            }
        )

    # 6) Stats + outputs
    df = pd.DataFrame(rand_rows)
    rand_test = df["test_r2"].values
    ge_count = int(np.sum(rand_test >= real_test_r2))
    p_value = float((ge_count + 1) / (N_ITERATIONS + 1))
    failed_count = int(df["failed"].sum())
    elapsed = time.time() - t0

    print("[Step 6/6] Summary")
    print("-" * 72)
    print(f"baseline test R2: {real_test_r2:.4f}")
    print(
        f"randomized test R2: mean={rand_test.mean():.4f}, std={rand_test.std():.4f}, "
        f"p95={np.quantile(rand_test, 0.95):.4f}, max={rand_test.max():.4f}"
    )
    print(f"p-value: {p_value:.4f} ({ge_count}/{N_ITERATIONS})")
    print(f"failed iterations: {failed_count}")
    print(f"elapsed: {elapsed:.1f}s")
    conclusion = "PASS - model signal is significant (p < 0.05)" if p_value < 0.05 else "FAIL - signal not significant"
    print(f"conclusion: {conclusion}")
    print("-" * 72)

    df["real_test_r2"] = real_test_r2
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.hist(rand_test, bins=25, color="#74c476", edgecolor="white", alpha=0.85, label="Randomized")
    plt.axvline(real_test_r2, color="#e74c3c", linewidth=2.5, linestyle="--", label=f"Real model (R2={real_test_r2:.4f})")
    plt.xlabel("Test R2")
    plt.ylabel("Count")
    plt.title(f"DNN Y-Randomization (p={p_value:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("DNN Y-Randomization Summary\n")
        f.write("=" * 72 + "\n")
        f.write(f"iterations={N_ITERATIONS}\n")
        f.write(f"baseline_test_r2={real_test_r2:.6f}\n")
        f.write(f"random_mean_test_r2={rand_test.mean():.6f}\n")
        f.write(f"random_std_test_r2={rand_test.std():.6f}\n")
        f.write(f"random_p95_test_r2={np.quantile(rand_test, 0.95):.6f}\n")
        f.write(f"random_max_test_r2={rand_test.max():.6f}\n")
        f.write(f"p_value={p_value:.6f}\n")
        f.write(f"ge_count={ge_count}\n")
        f.write(f"failed_iterations={failed_count}\n")
        f.write(f"elapsed_sec={elapsed:.2f}\n")
        f.write(f"conclusion={conclusion}\n")

    print(f"saved: {CSV_PATH}")
    print(f"saved: {PLOT_PATH}")
    print(f"saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
