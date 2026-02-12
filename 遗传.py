import pickle
import random
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn.utils
import sklearn.model_selection as _ms
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from feature_config import ALL_FEATURE_COLS, SELECTED_FEATURE_COLS, resolve_target_col

# ----------------------------------------------------------------------
# Compatibility shim for genetic_selection on newer scikit-learn
# ----------------------------------------------------------------------
if not hasattr(sklearn.utils, "_joblib"):
    sklearn.utils._joblib = joblib
    sys.modules["sklearn.utils._joblib"] = joblib

if not hasattr(_ms.cross_val_score, "_ga_compat"):
    _orig_cvs = _ms.cross_val_score

    def _cross_val_score_compat(*args, **kwargs):
        kwargs.pop("fit_params", None)
        return _orig_cvs(*args, **kwargs)

    _cross_val_score_compat._ga_compat = True
    _ms.cross_val_score = _cross_val_score_compat

from genetic_selection import GeneticSelectionCV


# =========================
# 配置区
# =========================
DATA_PATH = "data/molecular_features.xlsx"
MODEL_PATH = "results/ga_nonlinear_model.pkl"
SUMMARY_PATH = "results/ga_nonlinear_summary.txt"

# 可选: "all" (20特征) / "selected" (16特征)
FEATURE_MODE = "all"

# 搜索规模:
# - fast: 快速验证代码与方向（推荐先跑）
# - full: 更充分搜索，耗时显著增加
SEARCH_MODE = "fast"

# 数据划分比例: train/val/test = 60/20/20
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.25
RANDOM_STATE = 42

# 多 seed 评估，让结果更稳健
SEEDS_FAST = [42, 52]
SEEDS_FULL = [42, 49, 56, 63, 70]

# 解释性配置:
# 随机森林没有单一解析公式，因此额外拟合一个线性代理模型用于输出公式
EXPORT_SURROGATE_FORMULA = True
SURROGATE_ALPHA = 1.0


def choose_features(mode: str):
    """按配置选择输入特征集合。"""
    if mode == "all":
        return ALL_FEATURE_COLS
    if mode == "selected":
        return SELECTED_FEATURE_COLS
    raise ValueError("FEATURE_MODE 仅支持 'all' 或 'selected'")


def get_ga_search_space(mode: str):
    """
    返回 GA 参数候选集合。
    说明:
    - 为控制训练时间，使用离散候选集合而非无限连续空间。
    - full 模式下候选更丰富。
    """
    if mode == "fast":
        return [
            {
                "max_features": 8,
                "n_population": 60,
                "n_generations": 20,
                "crossover_proba": 0.7,
                "mutation_proba": 0.1,
                "mutation_independent_proba": 0.03,
                "tournament_size": 3,
                "n_gen_no_change": 8,
            },
            {
                "max_features": 10,
                "n_population": 80,
                "n_generations": 25,
                "crossover_proba": 0.7,
                "mutation_proba": 0.15,
                "mutation_independent_proba": 0.05,
                "tournament_size": 3,
                "n_gen_no_change": 8,
            },
        ]

    if mode == "full":
        return [
            {
                "max_features": 8,
                "n_population": 120,
                "n_generations": 35,
                "crossover_proba": 0.8,
                "mutation_proba": 0.1,
                "mutation_independent_proba": 0.03,
                "tournament_size": 3,
                "n_gen_no_change": 10,
            },
            {
                "max_features": 10,
                "n_population": 160,
                "n_generations": 40,
                "crossover_proba": 0.8,
                "mutation_proba": 0.15,
                "mutation_independent_proba": 0.05,
                "tournament_size": 4,
                "n_gen_no_change": 12,
            },
            {
                "max_features": 12,
                "n_population": 200,
                "n_generations": 50,
                "crossover_proba": 0.85,
                "mutation_proba": 0.2,
                "mutation_independent_proba": 0.06,
                "tournament_size": 4,
                "n_gen_no_change": 12,
            },
        ]

    raise ValueError("SEARCH_MODE 仅支持 'fast' 或 'full'")


def build_ga_base_estimator(seed: int):
    """
    GA 进化阶段使用的基础评估器（非线性、速度与稳定性平衡）。
    深度与树数略小，降低 GA 内部交叉验证成本。
    """
    return RandomForestRegressor(
        n_estimators=80,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


def build_final_estimator(seed: int):
    """
    最终预测模型（比 GA 阶段更强一些）。
    """
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )


def build_formula(feature_names, intercept, coefs):
    """将线性模型参数拼接为可读的经验公式。"""
    formula = f"χ ≈ {intercept:.6f}"
    for w, feat in zip(coefs, feature_names):
        sign = "+" if w >= 0 else "-"
        formula += f" {sign} {abs(w):.6f}*{feat}"
    return formula


def evaluate_single_config(X_train, y_train, X_val, y_val, feature_names, cfg, seed):
    """
    在单个 seed 下评估一组 GA 参数:
    1) 用 GA 在 train 上选特征
    2) 用最终非线性模型在选中特征上训练
    3) 在 val 上计算指标
    """
    # 旧版 genetic_selection 不支持 random_state，这里手动固定随机性
    random.seed(seed)
    np.random.seed(seed)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    ga_estimator = build_ga_base_estimator(seed)

    selector = GeneticSelectionCV(
        estimator=ga_estimator,
        cv=cv,
        scoring="r2",
        verbose=0,
        n_jobs=-1,
        caching=True,
        **cfg,
    )
    selector.fit(X_train, y_train)

    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)

    final_estimator = build_final_estimator(seed)
    final_estimator.fit(X_train_sel, y_train)
    y_pred = final_estimator.predict(X_val_sel)

    selected_features = [str(x) for x in np.array(feature_names)[selector.support_]]
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return {
        "seed": seed,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "selected_features": selected_features,
        "selector": selector,
        "model": final_estimator,
    }


def summarize_config(cfg):
    return (
        f"max_features={cfg['max_features']}, pop={cfg['n_population']}, gen={cfg['n_generations']}, "
        f"cross={cfg['crossover_proba']}, mut={cfg['mutation_proba']}, mut_ind={cfg['mutation_independent_proba']}, "
        f"tour={cfg['tournament_size']}, early_stop={cfg['n_gen_no_change']}"
    )


def main():
    # 1) 读取数据
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)
    feature_cols = choose_features(FEATURE_MODE)

    X = data[feature_cols].values
    y = data[target_col].values

    # 2) 三段划分，避免验证集和测试集信息泄漏
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE_IN_TRAINVAL,
        random_state=RANDOM_STATE,
    )

    seeds = SEEDS_FAST if SEARCH_MODE == "fast" else SEEDS_FULL
    ga_space = get_ga_search_space(SEARCH_MODE)

    print(
        f"模式: feature_mode={FEATURE_MODE}, search_mode={SEARCH_MODE}\n"
        f"样本: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
        f"特征数={len(feature_cols)}"
    )
    print("开始 GA 参数搜索（非线性 RF 版本）...")

    # 3) 外层参数搜索（每组参数多 seed 评估）
    search_rows = []
    best_cfg = None
    best_score = -np.inf
    best_seed_result = None

    for idx, cfg in enumerate(ga_space, start=1):
        print(f"\n[{idx}/{len(ga_space)}] 评估参数组: {summarize_config(cfg)}")
        seed_results = []
        for seed in seeds:
            result = evaluate_single_config(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=feature_cols,
                cfg=cfg,
                seed=seed,
            )
            seed_results.append(result)
            print(
                f"  seed={seed} | val R2={result['r2']:.4f}, "
                f"MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}, "
                f"n_feat={len(result['selected_features'])}"
            )

        mean_r2 = float(np.mean([r["r2"] for r in seed_results]))
        std_r2 = float(np.std([r["r2"] for r in seed_results]))
        mean_mae = float(np.mean([r["mae"] for r in seed_results]))
        mean_rmse = float(np.mean([r["rmse"] for r in seed_results]))

        row = {
            "cfg": cfg,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_mae": mean_mae,
            "mean_rmse": mean_rmse,
            "seed_results": seed_results,
        }
        search_rows.append(row)

        print(
            f"  => 参数组汇总: mean R2={mean_r2:.4f} ± {std_r2:.4f}, "
            f"mean MAE={mean_mae:.4f}, mean RMSE={mean_rmse:.4f}"
        )

        if mean_r2 > best_score:
            best_score = mean_r2
            best_cfg = cfg
            # 在该参数组里，按 val R2 选一个最佳 seed 结果用于后续 warm start 参考
            best_seed_result = max(seed_results, key=lambda x: x["r2"])

    print("\n" + "=" * 70)
    print("GA 搜索完成，最佳参数组：")
    print(summarize_config(best_cfg))
    print(f"验证集均值 R2: {best_score:.4f}")

    # 4) 用最佳参数在 train+val 上重新选特征，再在 test 上评估
    X_refit = np.vstack([X_train, X_val])
    y_refit = np.concatenate([y_train, y_val])

    refit_seed = best_seed_result["seed"]
    random.seed(refit_seed)
    np.random.seed(refit_seed)
    selector_final = GeneticSelectionCV(
        estimator=build_ga_base_estimator(refit_seed),
        cv=KFold(n_splits=5, shuffle=True, random_state=refit_seed),
        scoring="r2",
        verbose=0,
        n_jobs=-1,
        caching=True,
        **best_cfg,
    )
    selector_final.fit(X_refit, y_refit)

    X_refit_sel = selector_final.transform(X_refit)
    X_test_sel = selector_final.transform(X_test)

    final_model = build_final_estimator(refit_seed)
    final_model.fit(X_refit_sel, y_refit)
    y_test_pred = final_model.predict(X_test_sel)

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    selected_features = [str(x) for x in np.array(feature_cols)[selector_final.support_]]
    print("\n最终入选特征：")
    print(selected_features)
    print(
        f"\n最终 Test 指标: R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}"
    )

    # 4.1) 输出线性代理公式（用于解释，不替代主模型预测）
    surrogate_info = None
    if EXPORT_SURROGATE_FORMULA:
        surrogate = Ridge(alpha=SURROGATE_ALPHA, random_state=RANDOM_STATE)
        surrogate.fit(X_refit_sel, y_refit)
        y_sur_pred = surrogate.predict(X_test_sel)
        sur_r2 = r2_score(y_test, y_sur_pred)
        sur_mae = mean_absolute_error(y_test, y_sur_pred)
        sur_rmse = np.sqrt(mean_squared_error(y_test, y_sur_pred))
        sur_coefs = surrogate.coef_.ravel() if hasattr(surrogate.coef_, "ravel") else surrogate.coef_
        formula = build_formula(selected_features, float(surrogate.intercept_), sur_coefs)
        surrogate_info = {
            "model": "Ridge",
            "alpha": SURROGATE_ALPHA,
            "r2": float(sur_r2),
            "mae": float(sur_mae),
            "rmse": float(sur_rmse),
            "formula": formula,
        }
        print("\n线性代理公式（解释用）：")
        print(formula)
        print(
            f"线性代理 Test 指标: R2={sur_r2:.4f}, MAE={sur_mae:.4f}, RMSE={sur_rmse:.4f}"
        )

    # 5) 保存模型包和搜索摘要
    bundle = {
        "model": final_model,
        "selector": selector_final,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "selected_features": selected_features,
        "best_cfg": best_cfg,
        "search_mode": SEARCH_MODE,
        "feature_mode": FEATURE_MODE,
        "seed": refit_seed,
        "metrics_test": {"r2": test_r2, "mae": test_mae, "rmse": test_rmse},
        "surrogate_info": surrogate_info,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("GA 非线性特征选择 + RF 回归 结果摘要\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            f"feature_mode={FEATURE_MODE}, search_mode={SEARCH_MODE}, "
            f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}\n\n"
        )
        f.write("参数搜索结果:\n")
        for i, row in enumerate(search_rows, start=1):
            f.write(
                f"{i}. {summarize_config(row['cfg'])}\n"
                f"   mean_r2={row['mean_r2']:.4f} ± {row['std_r2']:.4f}, "
                f"mean_mae={row['mean_mae']:.4f}, mean_rmse={row['mean_rmse']:.4f}\n"
            )
        f.write("\n最佳参数:\n")
        f.write(summarize_config(best_cfg) + "\n")
        f.write(f"\nselected_features ({len(selected_features)}):\n")
        f.write(str(selected_features) + "\n")
        f.write(
            f"\nTest metrics: R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}\n"
        )
        if surrogate_info is not None:
            f.write(
                f"\nSurrogate ({surrogate_info['model']}, alpha={surrogate_info['alpha']}) "
                f"metrics: R2={surrogate_info['r2']:.4f}, "
                f"MAE={surrogate_info['mae']:.4f}, RMSE={surrogate_info['rmse']:.4f}\n"
            )
            f.write("Surrogate formula:\n")
            f.write(surrogate_info["formula"] + "\n")

    print(f"\n模型包已保存: {MODEL_PATH}")
    print(f"搜索摘要已保存: {SUMMARY_PATH}")


if __name__ == "__main__":
    freeze_support()
    main()
