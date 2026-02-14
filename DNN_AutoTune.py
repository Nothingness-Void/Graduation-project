# -*- coding: utf-8 -*-
"""
DNN_AutoTune.py
使用 Keras Tuner Hyperband 自动搜索 DNN 架构（无测试集泄漏版本）

改进:
1. 搜索空间: 1-3 层, 12-64 节点 (适配 18 特征)
2. 参数预算: ratio 上限 20
3. 增加 batch_size 搜索
4. 增加 Hyperband 迭代次数
5. y 标准化选项
6. 更多重训次数 (8 次)
"""

import os
import pickle
import numpy as np
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import layers, regularizers, optimizers, callbacks as keras_callbacks, losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import keras_tuner as kt
from tqdm.auto import tqdm
from feature_config import SELECTED_FEATURE_COLS, resolve_target_col

# =========================
# 配置区
# =========================
DATA_PATH = "data/molecular_features.xlsx"
MODEL_PATH = "results/best_model.keras"
PREPROCESS_PATH = "results/best_model_preprocess.pkl"
SUMMARY_PATH = "results/tuner_summary.txt"



# 三段划分: train/val/test = 60/20/20
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.25
RANDOM_STATE = 42

# 小数据集约束: 参数/样本比上限
MAX_PARAM_RATIO = 20.0

# y 标准化 (对 Huggins 参数右偏分布有帮助)
SCALE_Y = True

# 运行规模
MAX_EPOCHS = 200
HYPERBAND_FACTOR = 3
HYPERBAND_ITERATIONS = 2
RETRAIN_RUNS = 8





class ProgressHyperband(kt.Hyperband):
    """Hyperband 搜索阶段的 trial 进度条。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial_bar = None
        self._seen_trial_ids = set()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        if self._trial_bar is None:
            total_trials = getattr(self.oracle, "max_trials", None)
            self._trial_bar = tqdm(
                total=total_trials,
                desc="Hyperband Trials",
                unit="trial",
                dynamic_ncols=True,
            )

        if trial.trial_id not in self._seen_trial_ids:
            self._seen_trial_ids.add(trial.trial_id)
            self._trial_bar.update(1)

        return super().run_trial(trial, *fit_args, **fit_kwargs)

    def search(self, *args, **kwargs):
        try:
            return super().search(*args, **kwargs)
        finally:
            if self._trial_bar is not None:
                self._trial_bar.close()
                self._trial_bar = None
                self._seen_trial_ids.clear()


class BoundedHyperModel(kt.HyperModel):
    """带参数预算约束的 HyperModel。"""

    def __init__(self, n_features: int, n_train: int):
        self.n_features = n_features
        self.n_train = n_train

    def build(self, hp):
        model = keras.Sequential()

        n_layers = hp.Int("n_layers", min_value=1, max_value=3, step=1)
        use_bn = hp.Boolean("use_batchnorm", default=True)
        l2_val = hp.Choice("l2_reg", values=[0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

        for i in range(n_layers):
            units = hp.Choice(f"units_layer_{i}", values=[12, 16, 24, 32, 48, 64])
            dropout_rate = hp.Choice(f"dropout_layer_{i}", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
            if i == 0:
                model.add(
                    keras.layers.Dense(
                        units,
                        activation="relu",
                        input_shape=(self.n_features,),
                        kernel_regularizer=regularizers.l2(l2_val) if l2_val > 0 else None,
                    )
                )
            else:
                model.add(
                    keras.layers.Dense(
                        units,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(l2_val) if l2_val > 0 else None,
                    )
                )

            if use_bn:
                model.add(keras.layers.BatchNormalization())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(1, activation="linear"))

        lr = hp.Float("learning_rate", min_value=5e-5, max_value=5e-3, sampling="log")
        loss_name = hp.Choice("loss", values=["huber", "mae", "mse"])
        if loss_name == "huber":
            loss = keras.losses.Huber(delta=1.0)
        elif loss_name == "mae":
            loss = "mae"
        else:
            loss = "mse"

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
            metrics=["mae"],
        )
        return model

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # 参数超预算直接短路
        param_ratio = model.count_params() / max(1, self.n_train)
        if param_ratio > MAX_PARAM_RATIO:
            history = keras.callbacks.History()
            history.history["val_mae"] = [1e6]
            history.history["val_loss"] = [1e6]
            return history

        # 搜索 batch_size
        batch_size = hp.Choice("batch_size", values=[8, 16, 32, 64])
        kwargs["batch_size"] = batch_size

        return model.fit(x, y, validation_data=validation_data, callbacks=callbacks, **kwargs)


def main():
    # 1) 数据加载与特征选择
    data = pd.read_excel(DATA_PATH)
    feature_cols = SELECTED_FEATURE_COLS
    target_col = resolve_target_col(data.columns)
    X = data[feature_cols].values
    y = data[target_col].values

    # 2) 三段划分，避免测试集泄漏
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE_IN_TRAINVAL,
        random_state=RANDOM_STATE,
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = None
    y_train_fit = y_train
    y_val_fit = y_val
    if SCALE_Y:
        scaler_y = StandardScaler()
        y_train_fit = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_fit = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    n_features = X_train_scaled.shape[1]
    n_train = X_train_scaled.shape[0]
    print(
        f"样本划分: train={len(y_train)}, val={len(y_val)}, test={len(y_test)} | "
        f"features={n_features}, scale_y={SCALE_Y}"
    )

    hypermodel = BoundedHyperModel(n_features=n_features, n_train=n_train)

    tuner = ProgressHyperband(
        hypermodel=hypermodel,
        objective=kt.Objective("val_mae", direction="min"),
        max_epochs=MAX_EPOCHS,
        factor=HYPERBAND_FACTOR,
        hyperband_iterations=HYPERBAND_ITERATIONS,
        directory="tuner_logs",
        project_name="dnn_autotune_v3",
        overwrite=True,
    )

    callbacks = [
        keras_callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras_callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
    ]

    print("\n开始 Hyperband 搜索...")
    tuner.search(
        X_train_scaled,
        y_train_fit,
        validation_data=(X_val_scaled, y_val_fit),
        callbacks=callbacks,
        verbose=0,
    )

    top_hps = tuner.get_best_hyperparameters(num_trials=5)
    print("\nTop 5 架构:")
    summary_lines = []
    for rank, hp in enumerate(top_hps, 1):
        n_layers = hp.get("n_layers")
        layers_info = []
        for i in range(n_layers):
            units = hp.get(f"units_layer_{i}")
            dropout = hp.get(f"dropout_layer_{i}")
            layers_info.append(f"{units}(d={dropout})")
        arch_str = " -> ".join(layers_info) + " -> 1"
        built_model = tuner.hypermodel.build(hp)
        n_params = built_model.count_params()
        line = (
            f"#{rank}: [{arch_str}] BN={hp.get('use_batchnorm')} "
            f"L2={hp.get('l2_reg')} LR={hp.get('learning_rate'):.6f} "
            f"Loss={hp.get('loss')} BS={hp.get('batch_size')} "
            f"Params={n_params} "
            f"Ratio={n_params/max(1, n_train):.2f}"
        )
        print(line)
        summary_lines.append(line)

    # 3) 固定最优超参数，多种子重训并在测试集评估稳定性
    best_hp = top_hps[0]
    all_results = []
    best_model = None
    best_val_loss = float("inf")
    best_seed = None

    best_batch_size = best_hp.get("batch_size")

    print(f"\n使用最优架构重训 {RETRAIN_RUNS} 次...")
    for run in tqdm(range(RETRAIN_RUNS), desc="Retrain Runs", unit="run", dynamic_ncols=True):
        seed = 42 + run * 7
        keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        model = tuner.hypermodel.build(best_hp)
        history = model.fit(
            X_train_scaled,
            y_train_fit,
            validation_data=(X_val_scaled, y_val_fit),
            epochs=MAX_EPOCHS * 2,
            batch_size=best_batch_size,
            callbacks=[
                keras_callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
                keras_callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
            ],
            verbose=0,
        )

        y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        val_loss = float(np.min(history.history["val_loss"]))

        all_results.append(
            {"run": run + 1, "seed": seed, "val_loss": val_loss, "R2": r2, "MAE": mae, "RMSE": rmse}
        )
        tqdm.write(
            f"Run {run+1}: val_loss={val_loss:.4f}, test R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_seed = seed

    results_df = pd.DataFrame(all_results)
    print("\n==================== 重训统计 ====================")
    print(results_df.to_string(index=False))
    print(
        f"\nR2 mean={results_df['R2'].mean():.4f}, std={results_df['R2'].std():.4f}, "
        f"best={results_df['R2'].max():.4f}"
    )
    print(f"按 val_loss 选中的最佳 seed: {best_seed}, val_loss={best_val_loss:.4f}")

    # 4) 保存最优模型与预处理信息
    best_model.save(MODEL_PATH)
    with open(PREPROCESS_PATH, "wb") as f:
        pickle.dump(
            {
                "feature_cols": feature_cols,
                "target_col": target_col,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,

                "best_hp": best_hp.values,
                "best_seed": best_seed,
            },
            f,
        )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("DNN 自动架构搜索结果 (无测试集泄漏版本)\n")
        f.write(f"{'='*70}\n\n")
        f.write(
            f"数据: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
            f"features={n_features}, scale_y={SCALE_Y}\n"
        )
        f.write(f"参数约束: ratio <= {MAX_PARAM_RATIO}\n\n")
        f.write("Top 5 架构:\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n重训结果:\n")
        f.write(results_df.to_string(index=False) + "\n")
        f.write(
            f"\nR2 mean={results_df['R2'].mean():.4f}, std={results_df['R2'].std():.4f}, "
            f"best={results_df['R2'].max():.4f}\n"
        )
        f.write(f"best_seed={best_seed}, best_val_loss={best_val_loss:.4f}\n")
        f.write(f"\nBest HP:\n{best_hp.values}\n")

    print(f"\n最优模型已保存: {MODEL_PATH}")
    print(f"预处理信息已保存: {PREPROCESS_PATH}")
    print(f"搜索摘要已保存: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
