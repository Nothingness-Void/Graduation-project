# -*- coding: utf-8 -*-
"""
遗传.py - 遗传算法特征选择 (基于 DEAP)

从 ~320 维全量 RDKit 描述符中选出最优特征子集，
使用 RandomForest 作为评估器，5 折交叉验证 R² 作为适应度。

用法:
    python 遗传.py

输出:
    - results/ga_selected_features.txt    (选中的特征列表)
    - results/ga_evolution_log.csv        (每代进化日志)
    - results/ga_best_model.pkl           (最优模型)
    - feature_config.py                   (自动更新)
"""
import os
import random
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

# ========== 配置区 ==========
DATA_PATH = "data/molecular_features.xlsx"
RESULTS_DIR = "results"

# GA 参数 (已针对 ~320 特征 + ~1900 样本做了平衡)
POPULATION_SIZE = 100       # 种群大小
N_GENERATIONS = 60          # 最大进化代数
CROSSOVER_PROB = 0.7        # 交叉概率
MUTATION_PROB = 0.15        # 个体变异概率
MUTATION_IND_PROB = 0.03    # 每个基因位翻转概率
TOURNAMENT_SIZE = 3         # 锦标赛选择大小
EARLY_STOP_GENS = 12        # 连续多少代无改善则停止
MIN_FEATURES = 5            # 最少选择特征数
MAX_FEATURES = 40           # 最多选择特征数

# 评估器参数 (轻量 RF，平衡速度与精度)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 8
RF_N_JOBS = -1
CV_FOLDS = 3               # 交叉验证折数 (3折比5折快~40%)

# 数据划分
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)


# ========== 工具函数 ==========
def format_time(seconds):
    """格式化秒数为可读时间。"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def load_data():
    """加载特征矩阵，自动检测目标列。"""
    data = pd.read_excel(DATA_PATH)

    # 检测目标列 (只匹配精确的目标列名，避免匹配 RDKit 的 Chi 描述符)
    target_col = None
    for candidate in ["chi_result", "chi-result", "χ-result", "χ"]:
        if candidate in data.columns:
            target_col = candidate
            break
    if target_col is None:
        # 只匹配包含 'result' 的列（不匹配 'chi'，因为 RDKit 有 Chi 描述符）
        for c in data.columns:
            if "result" in str(c).lower():
                target_col = c
                break
    if target_col is None:
        raise KeyError(f"未找到目标列。现有列: {list(data.columns)[-5:]}")

    feature_cols = [c for c in data.columns if c != target_col]
    X = data[feature_cols].values.astype(np.float64)
    y = data[target_col].values.astype(np.float64)

    # 处理残留 NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    return X, y, feature_cols



def evaluate_individual(individual, X_train, y_train, base_estimator, cv_folds):
    """
    评估一个个体 (特征子集) 的适应度。

    返回: (cv_r2,)  注意: DEAP 要求返回元组
    """
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

    # 惩罚: 特征数不在合法范围内
    n_selected = len(selected_idx)
    if n_selected < MIN_FEATURES or n_selected > MAX_FEATURES:
        return (-1.0,)

    X_subset = X_train[:, selected_idx]

    try:
        scores = cross_val_score(
            base_estimator, X_subset, y_train,
            cv=cv_folds, scoring="r2", n_jobs=1
        )
        fitness = float(np.mean(scores))
    except Exception:
        fitness = -1.0

    return (fitness,)


# ========== DEAP 进化引擎 ==========
def setup_deap(n_features):
    """配置 DEAP 工具箱。"""
    # 清理可能的旧定义 (重复运行安全)
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # 初始化策略: 每个基因位有一定概率为 1
    # 目标是初始选中 ~20 个特征 (20/320 ≈ 6%)
    init_prob = min(0.1, MAX_FEATURES / n_features)
    toolbox.register("attr_bool", random.random)
    toolbox.register(
        "individual", tools.initRepeat,
        creator.Individual,
        lambda: 1 if random.random() < init_prob else 0,
        n=n_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 遗传算子
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_IND_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    return toolbox


def run_ga(X_train, y_train, feature_cols):
    """运行遗传算法特征选择。"""
    n_features = X_train.shape[1]
    print(f"\nGA 特征选择配置:")
    print(f"  特征维度: {n_features}")
    print(f"  种群: {POPULATION_SIZE}, 最大代数: {N_GENERATIONS}")
    print(f"  特征数约束: [{MIN_FEATURES}, {MAX_FEATURES}]")
    print(f"  评估器: RF(n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}), CV={CV_FOLDS}")
    print(f"  早停: 连续 {EARLY_STOP_GENS} 代无改善")

    toolbox = setup_deap(n_features)

    # RF 评估器
    base_rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=2,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
    )

    # 注册评估函数
    toolbox.register(
        "evaluate", evaluate_individual,
        X_train=X_train, y_train=y_train,
        base_estimator=base_rf, cv_folds=CV_FOLDS
    )

    # 初始化种群
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    pop = toolbox.population(n=POPULATION_SIZE)

    # 统计工具
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)

    hof = tools.HallOfFame(5)

    # ========== 进化循环 (带进度显示) ==========
    print(f"\n{'='*70}")
    print(f"{'代':>4} | {'最优R2':>8} | {'平均R2':>8} | {'标准差':>8} | {'特征数':>5} | {'耗时':>8} | 状态")
    print(f"{'='*70}")

    # 评估初代
    t0 = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    record = stats.compile(pop)
    best_n_features = sum(hof[0])
    elapsed = time.time() - t0
    print(
        f"{'init':>4} | {record['max']:>8.4f} | {record['avg']:>8.4f} | "
        f"{record['std']:>8.4f} | {best_n_features:>5} | {format_time(elapsed):>8} | 初始化"
    )

    # 进化日志
    log_data = [{
        "gen": 0, "max_r2": record["max"], "avg_r2": record["avg"],
        "std_r2": record["std"], "best_n_features": best_n_features,
        "elapsed_sec": elapsed
    }]

    best_ever_r2 = record["max"]
    no_improve_count = 0

    for gen in range(1, N_GENERATIONS + 1):
        t_gen = time.time()

        # 选择 + 交叉 + 变异
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估需要重新计算的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 精英策略: 保留上一代最优
        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop)
        best_n_features = sum(hof[0])
        elapsed = time.time() - t_gen

        # 早停检测
        if record["max"] > best_ever_r2 + 1e-5:
            best_ever_r2 = record["max"]
            no_improve_count = 0
            status = "* 改善"
        else:
            no_improve_count += 1
            status = f"  停滞 {no_improve_count}/{EARLY_STOP_GENS}"

        print(
            f"{gen:>4} | {record['max']:>8.4f} | {record['avg']:>8.4f} | "
            f"{record['std']:>8.4f} | {best_n_features:>5} | {format_time(elapsed):>8} | {status}"
        )

        log_data.append({
            "gen": gen, "max_r2": record["max"], "avg_r2": record["avg"],
            "std_r2": record["std"], "best_n_features": best_n_features,
            "elapsed_sec": elapsed
        })

        if no_improve_count >= EARLY_STOP_GENS:
            print(f"\n>> 早停触发: 连续 {EARLY_STOP_GENS} 代无改善")
            break

    total_time = sum(d["elapsed_sec"] for d in log_data)
    print(f"{'='*70}")
    print(f"进化完成: {gen} 代, 总耗时 {format_time(total_time)}")

    # 保存进化日志
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(f"{RESULTS_DIR}/ga_evolution_log.csv", index=False, encoding="utf-8-sig")

    return hof, log_df


# ========== 主流程 ==========
def main():
    print("=" * 70)
    print("遗传算法特征选择 (DEAP)")
    print("=" * 70)

    # 1) 加载数据
    print("\n加载数据...")
    X, y, feature_cols = load_data()
    print(f"  数据: {X.shape[0]} 样本 × {X.shape[1]} 特征")

    # 2) 划分 train/test (GA 只在 train 上搜索，test 最终评估)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  划分: train={len(y_train)}, test={len(y_test)}")

    # 3) GA 搜索
    hof, log_df = run_ga(X_train, y_train, feature_cols)

    # 4) 提取最优特征
    best_individual = hof[0]
    selected_idx = [i for i, bit in enumerate(best_individual) if bit == 1]
    selected_features = [feature_cols[i] for i in selected_idx]
    best_cv_r2 = best_individual.fitness.values[0]

    print(f"\n{'='*70}")
    print(f"最优特征子集 ({len(selected_features)} 个, CV R2={best_cv_r2:.4f}):")
    print(f"{'='*70}")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:>3}. {feat}")

    # 5) 用最优特征在测试集上评估
    print(f"\n测试集评估...")
    final_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    final_rf.fit(X_train_sel, y_train)
    y_pred = final_rf.predict(X_test_sel)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  Test R2:   {test_r2:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

    # 6) 保存结果
    # 6a) 特征列表
    feat_path = f"{RESULTS_DIR}/ga_selected_features.txt"
    with open(feat_path, "w", encoding="utf-8") as f:
        f.write(f"GA 特征选择结果\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"总特征数: {len(feature_cols)} -> 选中: {len(selected_features)}\n")
        f.write(f"CV R2: {best_cv_r2:.4f}\n")
        f.write(f"Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}\n\n")
        f.write(f"选中的特征:\n")
        for feat in selected_features:
            f.write(f"  {feat}\n")
        f.write(f"\nGA 配置:\n")
        f.write(f"  种群={POPULATION_SIZE}, 代数={log_df['gen'].max()}\n")
        f.write(f"  交叉={CROSSOVER_PROB}, 变异={MUTATION_PROB}\n")
        f.write(f"  评估器: RF(n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}), CV={CV_FOLDS}\n")
    print(f"\n特征列表已保存: {feat_path}")

    # 6b) 最优模型
    model_path = f"{RESULTS_DIR}/ga_best_model.pkl"
    joblib.dump({
        "model": final_rf,
        "selected_features": selected_features,
        "selected_idx": selected_idx,
        "cv_r2": best_cv_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }, model_path)
    print(f"最优模型已保存: {model_path}")

    # 6c) 自动更新 feature_config.py
    update_feature_config(feature_cols, selected_features)

    # 6d) Top 5 个体
    print(f"\nHall of Fame (Top 5):")
    for rank, ind in enumerate(hof, 1):
        n_feat = sum(ind)
        r2 = ind.fitness.values[0]
        print(f"  #{rank}: {n_feat} 特征, CV R2={r2:.4f}")

    print(f"\n完成! 接下来可以运行:")
    print(f"  python 特征筛选.py  # 进一步特征筛选")


def update_feature_config(all_features, selected_features):
    """自动更新 feature_config.py。"""
    config_text = "\n".join([
        '"""Unified feature configuration for training/validation scripts."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Iterable",
        "",
        "# Selected features (auto-updated by 遗传.py / 特征筛选.py)",
        "SELECTED_FEATURE_COLS = [",
        *[f'    "{feat}",' for feat in selected_features],
        "]",
        "",
        "",
        'def resolve_target_col(columns: Iterable[str], preferred: str = "chi_result") -> str:',
        '    """Resolve target column with fallback for encoding variations."""',
        "    cols = list(columns)",
        "    if preferred in cols:",
        "        return preferred",
        "",
        '    candidates = [c for c in cols if "result" in str(c).lower()]',
        "    if candidates:",
        "        return candidates[0]",
        "",
        '    raise KeyError("未找到目标列：chi_result（或包含 result 的列名）")',
        "",
    ])
    Path("feature_config.py").write_text(config_text, encoding="utf-8")
    print(f"feature_config.py 已自动更新 (SELECTED={len(selected_features)})")


if __name__ == "__main__":
    main()
