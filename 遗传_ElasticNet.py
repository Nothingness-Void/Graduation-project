# -*- coding: utf-8 -*-
"""
遗传_ElasticNet.py - 物理解释型 GA 特征选择

使用 ElasticNet (线性模型) 作为评估器进行遗传算法特征选择。
ElasticNet 对 chi = A + B/T 这类线性物理关系敏感，
选出的特征更具物理可解释性。

用法:
    python 遗传_ElasticNet.py

输出:
    - results/ga_elasticnet_selected_features.txt
    - results/ga_elasticnet_evolution_log.csv
    - results/ga_elasticnet_best_model.pkl

注意: 本脚本不会自动更新 feature_config.py，
      需要用户根据两版 GA 结果综合决定最终特征列表。
"""
import os
import random
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from deap import base, creator, tools

from feature_config import resolve_target_col

warnings.filterwarnings("ignore")

# ========== 配置区 ==========
DATA_PATH = "data/molecular_features.xlsx"
RESULTS_DIR = "results"
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"

# GA 参数
POPULATION_SIZE = 100
N_GENERATIONS = 60
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.15
MUTATION_IND_PROB = 0.03
TOURNAMENT_SIZE = 3
EARLY_STOP_GENS = 12
MIN_FEATURES = 5
MAX_FEATURES = 30           # 线性模型不宜太多特征

# ElasticNet 参数
EN_ALPHA = 0.1              # 正则化强度
EN_L1_RATIO = 0.5           # L1/L2 混合比 (0.5 = 均衡)
EN_MAX_ITER = 2000

CV_FOLDS = 3
MANDATORY_FEATURES = ["Inv_T"]  # 物理先验必选特征

# 数据划分
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 输出路径 (独立于性能版，避免互相覆盖)
OUTPUT_PREFIX = "ga_elasticnet"
FEAT_PATH = os.path.join(RESULTS_DIR, f"{OUTPUT_PREFIX}_selected_features.txt")
LOG_PATH = os.path.join(RESULTS_DIR, f"{OUTPUT_PREFIX}_evolution_log.csv")
MODEL_PATH = os.path.join(RESULTS_DIR, f"{OUTPUT_PREFIX}_best_model.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ========== 工具函数 ==========
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def load_data():
    """加载特征矩阵，使用统一的目标列检测。"""
    data = pd.read_excel(DATA_PATH)
    target_col = resolve_target_col(data.columns)

    feature_cols = [c for c in data.columns if c != target_col]
    X = data[feature_cols].values.astype(np.float64)
    y = data[target_col].values.astype(np.float64)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    return X, y, feature_cols


def resolve_mandatory_indices(feature_cols):
    indices = []
    missing = []
    for feat in MANDATORY_FEATURES:
        if feat in feature_cols:
            indices.append(feature_cols.index(feat))
        else:
            missing.append(feat)
    if missing:
        raise KeyError(f"必选特征不存在于特征矩阵中: {missing}")
    return sorted(set(indices))


def repair_individual(individual, mandatory_idx):
    for idx in mandatory_idx:
        individual[idx] = 1
    return individual


def build_estimator():
    """构建 ElasticNet Pipeline (带 StandardScaler)。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=EN_ALPHA,
            l1_ratio=EN_L1_RATIO,
            max_iter=EN_MAX_ITER,
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate_individual(individual, X_train, y_train, base_estimator, cv_folds, mandatory_idx):
    repair_individual(individual, mandatory_idx)
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

    n_selected = len(selected_idx)
    if n_selected < MIN_FEATURES or n_selected > MAX_FEATURES:
        return (-1.0,)

    X_subset = X_train[:, selected_idx]

    try:
        scores = cross_val_score(
            base_estimator, X_subset, y_train,
            cv=cv_folds, scoring="r2", n_jobs=-1
        )
        fitness = float(np.mean(scores))
    except Exception:
        fitness = -1.0

    return (fitness,)


# ========== DEAP 进化引擎 ==========
def setup_deap(n_features, mandatory_idx):
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    init_prob = min(0.1, MAX_FEATURES / n_features)

    def init_individual():
        ind = creator.Individual(
            [1 if random.random() < init_prob else 0 for _ in range(n_features)]
        )
        repair_individual(ind, mandatory_idx)
        return ind

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_IND_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    return toolbox


def run_ga(X_train, y_train, feature_cols):
    n_features = X_train.shape[1]
    mandatory_idx = resolve_mandatory_indices(feature_cols)
    mandatory_names = [feature_cols[i] for i in mandatory_idx]
    print(f"\nGA-ElasticNet 特征选择配置:")
    print(f"  特征维度: {n_features}")
    print(f"  种群: {POPULATION_SIZE}, 最大代数: {N_GENERATIONS}")
    print(f"  特征数约束: [{MIN_FEATURES}, {MAX_FEATURES}]")
    print(f"  评估器: ElasticNet(alpha={EN_ALPHA}, l1={EN_L1_RATIO}), CV={CV_FOLDS}")
    print(f"  早停: 连续 {EARLY_STOP_GENS} 代无改善")
    print(f"  必选特征: {mandatory_names}")

    toolbox = setup_deap(n_features, mandatory_idx)

    base_estimator = build_estimator()

    toolbox.register(
        "evaluate", evaluate_individual,
        X_train=X_train, y_train=y_train,
        base_estimator=base_estimator, cv_folds=CV_FOLDS, mandatory_idx=mandatory_idx
    )

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    pop = toolbox.population(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)

    hof = tools.HallOfFame(5)

    # ========== 进化循环 ==========
    print(f"\n{'='*70}")
    print(f"{'代':>4} | {'最优R2':>8} | {'平均R2':>8} | {'标准差':>8} | {'特征数':>5} | {'耗时':>8} | 状态")
    print(f"{'='*70}")

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

    log_data = [{
        "gen": 0, "max_r2": record["max"], "avg_r2": record["avg"],
        "std_r2": record["std"], "best_n_features": best_n_features,
        "elapsed_sec": elapsed
    }]

    best_ever_r2 = record["max"]
    no_improve_count = 0

    for gen in range(1, N_GENERATIONS + 1):
        t_gen = time.time()

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                repair_individual(child1, mandatory_idx)
                repair_individual(child2, mandatory_idx)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                repair_individual(mutant, mandatory_idx)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop)
        best_n_features = sum(hof[0])
        elapsed = time.time() - t_gen

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

    log_df = pd.DataFrame(log_data)
    log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    return hof, log_df


# ========== 主流程 ==========
def main():
    print("=" * 70)
    print("遗传算法特征选择 — 物理解释版 (ElasticNet)")
    print("=" * 70)

    # 1) 加载数据
    print("\n加载数据...")
    X, y, feature_cols = load_data()
    print(f"  数据: {X.shape[0]} 样本 x {X.shape[1]} 特征")

    # 2) 复用已有 split（与性能版保持一致）
    if os.path.exists(SPLIT_INDEX_PATH):
        saved = np.load(SPLIT_INDEX_PATH)
        train_idx = saved["train_idx"]
        test_idx = saved["test_idx"]
        print(f"  已复用 split 索引: {SPLIT_INDEX_PATH}")
    else:
        all_indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
        np.savez(
            SPLIT_INDEX_PATH,
            train_idx=train_idx,
            test_idx=test_idx,
            n_samples=np.array([len(y)], dtype=np.int64),
            test_size=np.array([TEST_SIZE], dtype=np.float64),
            random_state=np.array([RANDOM_STATE], dtype=np.int64),
        )
        print(f"  split 索引已保存: {SPLIT_INDEX_PATH}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
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

    # 5) 测试集评估 (用 ElasticNet)
    print(f"\n测试集评估 (ElasticNet)...")
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=EN_ALPHA,
            l1_ratio=EN_L1_RATIO,
            max_iter=EN_MAX_ITER,
            random_state=RANDOM_STATE,
        )),
    ])
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    final_model.fit(X_train_sel, y_train)
    y_pred = final_model.predict(X_test_sel)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  Test R2:   {test_r2:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

    # 6) ElasticNet 系数分析 (物理解释性)
    en_model = final_model.named_steps["model"]
    coefs = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": en_model.coef_,
        "Abs_Coef": np.abs(en_model.coef_),
    }).sort_values("Abs_Coef", ascending=False)

    print(f"\n{'='*70}")
    print(f"ElasticNet 系数 (物理解释性排名):")
    print(f"{'='*70}")
    print(f"  Intercept = {en_model.intercept_:.4f}")
    for _, row in coefs.iterrows():
        sign = "+" if row["Coefficient"] > 0 else "-"
        print(f"  {sign} {row['Abs_Coef']:.4f}  {row['Feature']}")

    # 7) 保存结果
    with open(FEAT_PATH, "w", encoding="utf-8") as f:
        f.write(f"GA-ElasticNet 特征选择结果 (物理解释版)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"总特征数: {len(feature_cols)} -> 选中: {len(selected_features)}\n")
        f.write(f"CV R2: {best_cv_r2:.4f}\n")
        f.write(f"Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}\n\n")
        f.write(f"选中的特征:\n")
        for feat in selected_features:
            f.write(f"  {feat}\n")
        f.write(f"\nElasticNet 系数:\n")
        f.write(f"  Intercept = {en_model.intercept_:.4f}\n")
        for _, row in coefs.iterrows():
            sign = "+" if row["Coefficient"] > 0 else "-"
            f.write(f"  {sign} {row['Abs_Coef']:.4f}  {row['Feature']}\n")
        f.write(f"\nGA 配置:\n")
        f.write(f"  种群={POPULATION_SIZE}, 代数={log_df['gen'].max()}\n")
        f.write(f"  交叉={CROSSOVER_PROB}, 变异={MUTATION_PROB}\n")
        f.write(f"  评估器: ElasticNet(alpha={EN_ALPHA}, l1={EN_L1_RATIO}), CV={CV_FOLDS}\n")
        f.write(f"  必选特征: {MANDATORY_FEATURES}\n")
    print(f"\n特征列表已保存: {FEAT_PATH}")

    joblib.dump({
        "model": final_model,
        "selected_features": selected_features,
        "selected_idx": selected_idx,
        "cv_r2": best_cv_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "coefficients": coefs.to_dict("records"),
    }, MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")

    # Hall of Fame
    print(f"\nHall of Fame (Top 5):")
    for rank, ind in enumerate(hof, 1):
        n_feat = sum(ind)
        r2 = ind.fitness.values[0]
        print(f"  #{rank}: {n_feat} 特征, CV R2={r2:.4f}")

    print(f"\n完成!")
    print(f"注意: 本脚本不自动更新 feature_config.py")
    print(f"      请对比 RF 版结果后综合决定最终特征列表")


if __name__ == "__main__":
    main()
