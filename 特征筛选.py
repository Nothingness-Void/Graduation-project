# -*- coding: utf-8 -*-
"""
特征筛选.py - 综合特征筛选脚本

方法：
  1. 随机森林特征重要性排序
  2. 递归特征消除（RFECV）— 自动找最优特征数量
  3. 相关性过滤 — 去除高度共线特征
  4. 最终输出优化后的特征列表和数据

输出：
  - results/feature_selection.png   — 可视化图
  - results/feature_ranking.txt     — 特征排名详情
  - data/features_optimized.xlsx    — 筛选后的数据（可直接用于建模）
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
import warnings

from feature_config import SELECTED_FEATURE_COLS, resolve_target_col
from utils.data_utils import load_saved_split_indices

warnings.filterwarnings('ignore')

TEST_SIZE = 0.2
RANDOM_STATE = 42
SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"


def write_feature_config(selected_features, all_features, output_path='feature_config.py'):
    config_text = '\n'.join([
        '"""Unified feature configuration for training/validation scripts."""',
        '',
        'from __future__ import annotations',
        '',
        'from typing import Iterable',
        '',
        '# Selected features (auto-updated by 遗传.py / 特征筛选.py)',
        'SELECTED_FEATURE_COLS = [',
        *[f'    "{feat}",' for feat in selected_features],
        ']',
        '',
        '',
        'def resolve_target_col(columns: Iterable[str], preferred: str = "chi_result") -> str:',
        '    """Resolve target column with fallback for encoding variations."""',
        '    cols = list(columns)',
        '    if preferred in cols:',
        '        return preferred',
        '',
        '    candidates = [c for c in cols if "result" in str(c).lower()]',
        '    if candidates:',
        '        return candidates[0]',
        '',
        '    raise KeyError("未找到目标列：chi_result（或包含 result 的列名）")',
        '',
    ])
    Path(output_path).write_text(config_text, encoding='utf-8')
    print(f"统一特征配置已自动更新: {output_path}")


def main():
    # ========== 1. 数据加载 ==========
    df = pd.read_excel('data/molecular_features.xlsx')
    target_col = resolve_target_col(df.columns)

    # 从 feature_config.py 加载 GA 选出的特征（两阶段筛选: GA粗筛 → RFECV精筛）
    valid_cols = [c for c in SELECTED_FEATURE_COLS if c in df.columns]
    if len(valid_cols) >= 5:
        feature_cols = valid_cols
        print(f"从 feature_config.py 加载 GA 预选特征: {len(feature_cols)} 个")
    else:
        raise RuntimeError(
            f"feature_config.py 中的有效特征不足 ({len(valid_cols)} 个)。\n"
            f"请先运行 python 遗传.py 进行 GA 粗筛。"
        )

    X_all = df[feature_cols]
    y_all = df[target_col]

    # 关键：优先加载 遗传.py 保存的 split 索引，保证全链路一致
    split_result = load_saved_split_indices(len(df), SPLIT_INDEX_PATH)
    if split_result is not None:
        train_idx, test_idx = split_result
        print(f"已加载 遗传.py 保存的 split 索引，复用相同 train/test 划分。")
    else:
        all_indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
        Path("results").mkdir(parents=True, exist_ok=True)
        np.savez(
            SPLIT_INDEX_PATH,
            train_idx=train_idx,
            test_idx=test_idx,
            n_samples=np.array([len(df)], dtype=np.int64),
            test_size=np.array([TEST_SIZE], dtype=np.float64),
            random_state=np.array([RANDOM_STATE], dtype=np.int64),
        )
        print(f"未找到已保存的 split 索引，已自行划分并保存: {SPLIT_INDEX_PATH}")

    X_train = X_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    y_train = y_all.iloc[train_idx]
    y_test = y_all.iloc[test_idx]
    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)

    print(f"原始特征数: {len(feature_cols)}")
    print(f"样本数(训练集): {len(y)} | 测试集: {len(y_test)}")

    # ========== 1.1 目标值分布诊断 ==========
    print("\n" + "="*60)
    print("Step 0: 目标值分布诊断")
    print("="*60)
    print(y.describe())
    q = y.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    print("\n关键分位数:")
    print(q)

    # ========== 2. 随机森林特征重要性 ==========
    print("\n" + "="*60)
    print("Step 1: 随机森林特征重要性排序")
    print("="*60)

    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n全部特征重要性排名：")
    for rank, idx in enumerate(sorted_idx, 1):
        bar = "█" * int(importances[idx] * 100)
        print(f"  {rank:2d}. {feature_cols[idx]:30s}  {importances[idx]:.4f}  {bar}")

    # ========== 3. 相关性分析 ==========
    print("\n" + "="*60)
    print("Step 2: 高相关性特征检测 (|r| > 0.9)")
    print("="*60)

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > 0.9:
                high_corr_pairs.append((row, col, upper.loc[row, col]))

    if high_corr_pairs:
        print(f"\n发现 {len(high_corr_pairs)} 对高相关性特征：")
        for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"  {f1} ↔ {f2}: r={corr:.3f}")
    else:
        print("  未发现 |r| > 0.9 的特征对")

    # ========== 4. RFECV 递归特征消除 ==========
    print("\n" + "="*60)
    print("Step 3: RFECV 递归特征消除（自动确定最优特征数）")
    print("="*60)
    print("  正在训练集上运行 5 折交叉验证...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf_shuffle = KFold(n_splits=5, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        step=1,
        cv=kf_shuffle,
        scoring='r2',
        min_features_to_select=5,
        n_jobs=-1
    )
    rfecv.fit(X_scaled, y)

    print(f"\n  最优特征数量: {rfecv.n_features_}")
    print(f"  最优 CV R²: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - 5]:.4f}")

    selected_mask = rfecv.support_
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
    eliminated_features = [f for f, s in zip(feature_cols, selected_mask) if not s]

    print(f"\n  ✅ 保留的特征 ({len(selected_features)}):")
    for f in selected_features:
        rf_rank = sorted_idx.tolist().index(feature_cols.index(f)) + 1
        print(f"     {f} (RF重要性排名: #{rf_rank})")

    print(f"\n  ❌ 淘汰的特征 ({len(eliminated_features)}):")
    for f in eliminated_features:
        rf_rank = sorted_idx.tolist().index(feature_cols.index(f)) + 1
        print(f"     {f} (RF重要性排名: #{rf_rank})")

    # ========== 5. 最终验证：对比全特征 vs 筛选后 ==========
    print("\n" + "="*60)
    print("Step 4: 交叉验证对比")
    print("="*60)

    rf_full = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores_full = cross_val_score(rf_full, X_scaled, y, cv=rkf, scoring='r2')

    X_selected = X_scaled[:, selected_mask]
    rf_sel = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    scores_sel = cross_val_score(rf_sel, X_selected, y, cv=rkf, scoring='r2')

    print(f"\n  全部 {len(feature_cols)} 特征 → CV R²: {scores_full.mean():.4f} ± {scores_full.std():.4f}")
    print(f"  筛选 {len(selected_features)} 特征 → CV R²: {scores_sel.mean():.4f} ± {scores_sel.std():.4f}")

    # ========== 5.1 交叉验证诊断 ==========
    print("\n" + "="*60)
    print("Step 4.1: 交叉验证诊断 (每折 R² + 基线)")
    print("="*60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    full_fold_scores = []
    sel_fold_scores = []
    dummy_fold_scores = []

    for fold, (cv_train_idx, cv_test_idx) in enumerate(kf.split(X_scaled), 1):
        X_tr, X_te = X_scaled[cv_train_idx], X_scaled[cv_test_idx]
        y_tr, y_te = y[cv_train_idx], y[cv_test_idx]

        rf_full = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_full.fit(X_tr, y_tr)
        y_pred_full = rf_full.predict(X_te)
        r2_full = r2_score(y_te, y_pred_full)
        full_fold_scores.append(r2_full)

        rf_sel = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_sel.fit(X_tr[:, selected_mask], y_tr)
        y_pred_sel = rf_sel.predict(X_te[:, selected_mask])
        r2_sel = r2_score(y_te, y_pred_sel)
        sel_fold_scores.append(r2_sel)

        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_tr, y_tr)
        y_pred_dummy = dummy.predict(X_te)
        r2_dummy = r2_score(y_te, y_pred_dummy)
        dummy_fold_scores.append(r2_dummy)

        print(f"  Fold {fold}: Full={r2_full:.4f}, Selected={r2_sel:.4f}, Dummy={r2_dummy:.4f}")

    print("\n  每折 R² 统计:")
    print(f"  Full     mean={np.mean(full_fold_scores):.4f}, std={np.std(full_fold_scores):.4f}")
    print(f"  Selected mean={np.mean(sel_fold_scores):.4f}, std={np.std(sel_fold_scores):.4f}")
    print(f"  Dummy    mean={np.mean(dummy_fold_scores):.4f}, std={np.std(dummy_fold_scores):.4f}")

    if scores_sel.mean() >= scores_full.mean():
        print(f"\n  ✅ 筛选后性能 ≥ 全特征，推荐使用 {len(selected_features)} 个特征")
    else:
        diff = scores_full.mean() - scores_sel.mean()
        if diff < 0.01:
            print(f"\n  ✅ 筛选后性能仅下降 {diff:.4f}，但特征更少更稳定，推荐使用")
        else:
            print(f"\n  ⚠️ 筛选后性能下降 {diff:.4f}，请斟酌取舍")

    # ========== 6. 可视化 ==========
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 图1：特征重要性柱状图
    ax1 = axes[0]
    ax1.barh(range(len(sorted_idx)), importances[sorted_idx[::-1]], align='center')
    ax1.set_yticks(range(len(sorted_idx)))
    ax1.set_yticklabels([feature_cols[i] for i in sorted_idx[::-1]], fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title('Random Forest Feature Importance')

    # 图2：RFECV 曲线
    ax2 = axes[1]
    n_features_range = range(5, 5 + len(rfecv.cv_results_['mean_test_score']))
    ax2.plot(n_features_range, rfecv.cv_results_['mean_test_score'], 'b-o', markersize=3)
    ax2.fill_between(n_features_range,
                     np.array(rfecv.cv_results_['mean_test_score']) - np.array(rfecv.cv_results_['std_test_score']),
                     np.array(rfecv.cv_results_['mean_test_score']) + np.array(rfecv.cv_results_['std_test_score']),
                     alpha=0.2)
    ax2.axvline(x=rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_}')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('CV R²')
    ax2.set_title('RFECV: Optimal Number of Features')
    ax2.legend()

    # 图3：对比柱状图
    ax3 = axes[2]
    bars = ax3.bar([f'All Features\n({len(feature_cols)})', f'Selected\n({len(selected_features)})'],
                   [scores_full.mean(), scores_sel.mean()],
                   yerr=[scores_full.std(), scores_sel.std()],
                   color=['#3498db', '#2ecc71'], capsize=10)
    ax3.set_ylabel('CV R²')
    ax3.set_title('Performance Comparison')

    plt.tight_layout()
    plt.savefig('results/feature_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存至 results/feature_selection.png")

    # ========== 7. 保存结果 ==========
    # 保存筛选后的数据
    optimized_cols = selected_features + [target_col]
    df_optimized = df[optimized_cols]
    df_optimized.to_excel('data/features_optimized.xlsx', index=False)
    print(f"筛选后数据已保存至 data/features_optimized.xlsx ({len(selected_features)} 特征 + 目标值)")

    # 自动同步统一特征配置
    write_feature_config(selected_features, feature_cols, output_path='feature_config.py')

    # 保存排名详情
    with open('results/feature_ranking.txt', 'w', encoding='utf-8') as f:
        f.write("特征筛选结果\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"原始特征数: {len(feature_cols)}\n")
        f.write(f"筛选使用训练集样本数: {len(y)} (test_size={TEST_SIZE})\n")
        f.write(f"筛选后特征数: {len(selected_features)}\n\n")
        f.write("保留的特征:\n")
        for feat in selected_features:
            f.write(f"  {feat}\n")
        f.write(f"\n淘汰的特征:\n")
        for feat in eliminated_features:
            f.write(f"  {feat}\n")
        f.write(f"\nCV R² 对比:\n")
        f.write(f"  全特征: {scores_full.mean():.4f} ± {scores_full.std():.4f}\n")
        f.write(f"  筛选后: {scores_sel.mean():.4f} ± {scores_sel.std():.4f}\n")

        # 输出可直接复制的 Python 列表
        f.write(f"\n可直接复制到代码中的特征列表:\n")
        f.write(f"feature_cols = {selected_features}\n")

    print(f"排名详情已保存至 results/feature_ranking.txt")
    print(f"\n最终推荐特征列表 (可直接复制到模型代码中):")
    print(f"feature_cols = {selected_features}")


if __name__ == "__main__":
    main()
