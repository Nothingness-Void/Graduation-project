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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 数据加载 ==========
df = pd.read_excel('data/molecular_features.xlsx')

# 所有特征列（排除目标列）
feature_cols = ['MolWt1', 'logP1', 'TPSA1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'MaxAbsPartialCharge1', 'LabuteASA1',
                'CalcSpherocityIndex1','CalcRadiusOfGyration1',
                'MolWt2', 'logP2', 'TPSA2', 
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'MaxAbsPartialCharge2', 'LabuteASA2',
                'CalcSpherocityIndex2','CalcRadiusOfGyration2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity',
                'Delta_LogP', 'Delta_TPSA', 'HB_Match', 'Delta_MolMR', 'CSP3_1', 'CSP3_2', 'Inv_T']

X = df[feature_cols]
y = df['χ-result']

print(f"原始特征数: {len(feature_cols)}")
print(f"样本数: {len(y)}")

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
print("  正在运行 5 折交叉验证...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rfecv = RFECV(
    estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    step=1,
    cv=5,
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
scores_full = cross_val_score(rf_full, X_scaled, y, cv=5, scoring='r2')

X_selected = X_scaled[:, selected_mask]
rf_sel = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
scores_sel = cross_val_score(rf_sel, X_selected, y, cv=5, scoring='r2')

print(f"\n  全部 {len(feature_cols)} 特征 → CV R²: {scores_full.mean():.4f} ± {scores_full.std():.4f}")
print(f"  筛选 {len(selected_features)} 特征 → CV R²: {scores_sel.mean():.4f} ± {scores_sel.std():.4f}")

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
bars = ax3.bar(['All Features\n(35)', f'Selected\n({len(selected_features)})'],
               [scores_full.mean(), scores_sel.mean()],
               yerr=[scores_full.std(), scores_sel.std()],
               color=['#3498db', '#2ecc71'], capsize=10)
ax3.set_ylabel('CV R²')
ax3.set_title('Performance Comparison')

plt.tight_layout()
plt.savefig('results/feature_selection.png', dpi=300, bbox_inches='tight')
print(f"\n可视化已保存至 results/feature_selection.png")

# ========== 7. 保存结果 ==========
# 保存筛选后的数据
optimized_cols = selected_features + ['χ-result']
df_optimized = df[optimized_cols]
df_optimized.to_excel('data/features_optimized.xlsx', index=False)
print(f"筛选后数据已保存至 data/features_optimized.xlsx ({len(selected_features)} 特征 + 目标值)")

# 保存排名详情
with open('results/feature_ranking.txt', 'w', encoding='utf-8') as f:
    f.write("特征筛选结果\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"原始特征数: {len(feature_cols)}\n")
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
