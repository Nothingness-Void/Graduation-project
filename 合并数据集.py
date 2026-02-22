# -*- coding: utf-8 -*-
"""
合并数据集.py - 将旧数据与新数据合并为统一格式

旧数据: data/huggins_preprocessed.xlsx (323行, 含已处理的χ值)
新数据: data/43579_2022_237_MOESM1_ESM.csv (1586行)

输出: data/merged_dataset.csv
统一列: Polymer, Solvent, Polymer_SMILES, Solvent_SMILES, chi, temperature, source
"""
import pandas as pd
import numpy as np

# ========== 1. 加载旧数据 ==========
print("加载旧数据...")
df_old = pd.read_excel("data/huggins_preprocessed.xlsx")
print(f"  旧数据: {df_old.shape[0]} 行, 列: {list(df_old.columns)}")

# 映射旧数据列名到统一格式
df_old_mapped = pd.DataFrame({
    "Polymer": df_old["Compound 1"],
    "Solvent": df_old["Compound 2"],
    "Polymer_SMILES": df_old["SMILES 1"],
    "Solvent_SMILES": df_old["SMILES 2"],
    "chi": df_old["χ"].astype(float),
    "temperature": df_old["Measured at T (K)"].astype(float),
    "source": "old_dataset",
})

# ========== 2. 加载新数据 ==========
print("加载新数据...")
df_new = pd.read_csv("data/43579_2022_237_MOESM1_ESM.csv")
print(f"  新数据: {df_new.shape[0]} 行, 列: {list(df_new.columns)}")

df_new_mapped = pd.DataFrame({
    "Polymer": df_new["Polymer"],
    "Solvent": df_new["Solvent"],
    "Polymer_SMILES": df_new["Polymer SMILES"],
    "Solvent_SMILES": df_new["Solvent SMILES"],
    "chi": df_new["chi"].astype(float),
    "temperature": df_new["temperature"].astype(float),
    "source": "new_dataset",
})

# ========== 3. 合并 ==========
print("\n合并数据集...")
df_merged = pd.concat([df_old_mapped, df_new_mapped], ignore_index=True)
print(f"  合并后总行数: {df_merged.shape[0]}")

# ========== 4. 清洗 ==========

# 4.1 删除 SMILES 缺失的行
before = len(df_merged)
df_merged = df_merged.dropna(subset=["Polymer_SMILES", "Solvent_SMILES"])
dropped_smiles = before - len(df_merged)
if dropped_smiles > 0:
    print(f"  删除 SMILES 缺失: {dropped_smiles} 行")

# 4.1b 清除 SMILES 中的空白字符 (原始数据中存在 Tab 等混入)
df_merged["Polymer_SMILES"] = df_merged["Polymer_SMILES"].str.strip()
df_merged["Solvent_SMILES"] = df_merged["Solvent_SMILES"].str.strip()

# 4.1c 删除 strip 后变为空字符串的行
before = len(df_merged)
df_merged = df_merged[
    (df_merged["Polymer_SMILES"] != "") & (df_merged["Solvent_SMILES"] != "")
]
dropped_empty = before - len(df_merged)
if dropped_empty > 0:
    print(f"  删除空 SMILES (strip 后): {dropped_empty} 行")

# 4.2 删除 chi 缺失或无穷的行
before = len(df_merged)
df_merged = df_merged[np.isfinite(df_merged["chi"])]
dropped_chi = before - len(df_merged)
if dropped_chi > 0:
    print(f"  删除 chi 异常值: {dropped_chi} 行")

# 4.3 删除温度缺失的行
before = len(df_merged)
df_merged = df_merged.dropna(subset=["temperature"])
dropped_temp = before - len(df_merged)
if dropped_temp > 0:
    print(f"  删除温度缺失: {dropped_temp} 行")

# 4.4 去除精确重复 (同一聚合物SMILES + 溶剂SMILES + chi + temperature)
before = len(df_merged)
df_merged = df_merged.drop_duplicates(
    subset=["Polymer_SMILES", "Solvent_SMILES", "chi", "temperature"],
    keep="first"
)
dropped_dup = before - len(df_merged)
if dropped_dup > 0:
    print(f"  删除精确重复: {dropped_dup} 行")

# 4.5 解决 chi 冲突: 仅对同一 (SMILES对 + 温度) 存在不同 chi 值的组取中位数
group_keys = ["Polymer_SMILES", "Solvent_SMILES", "temperature"]
before = len(df_merged)

# 标记冲突组
chi_counts = df_merged.groupby(group_keys)["chi"].transform("nunique")
is_conflict = chi_counts > 1
n_conflicts = df_merged.loc[is_conflict, group_keys].drop_duplicates().shape[0]

if n_conflicts > 0:
    # 拆分: 冲突行 vs 非冲突行
    df_ok = df_merged[~is_conflict].copy()
    df_conflict = df_merged[is_conflict].copy()

    # 仅对冲突组做中位数聚合
    agg_dict = {c: "first" for c in df_conflict.columns if c not in group_keys + ["chi"]}
    agg_dict["chi"] = "median"
    df_resolved = df_conflict.groupby(group_keys, as_index=False).agg(agg_dict)

    # 合并回来
    df_merged = pd.concat([df_ok, df_resolved], ignore_index=True)
    after = len(df_merged)
    print(f"  解决 chi 冲突 (取中位数): {n_conflicts} 组, {before} → {after} 行")

# 4.6 过滤极端异常值 (|chi| > 10)
before = len(df_merged)
df_merged = df_merged[df_merged["chi"].abs() <= 10]
dropped_outlier = before - len(df_merged)
if dropped_outlier > 0:
    print(f"  删除极端异常值 (|chi|>10): {dropped_outlier} 行")

df_merged = df_merged.reset_index(drop=True)

# ========== 5. 统计 ==========
print(f"\n{'='*60}")
print(f"最终合并数据集统计")
print(f"{'='*60}")
print(f"总行数: {len(df_merged)}")
print(f"来源分布:")
print(df_merged["source"].value_counts().to_string())
print(f"\nchi 分布:")
print(df_merged["chi"].describe().to_string())
print(f"\ntemperature 分布:")
print(df_merged["temperature"].describe().to_string())
print(f"\n唯一聚合物: {df_merged['Polymer'].nunique()}")
print(f"唯一溶剂: {df_merged['Solvent'].nunique()}")
print(f"唯一聚合物 SMILES: {df_merged['Polymer_SMILES'].nunique()}")
print(f"唯一溶剂 SMILES: {df_merged['Solvent_SMILES'].nunique()}")

# ========== 6. 保存 ==========
output_path = "data/merged_dataset.csv"
df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n合并数据集已保存至: {output_path}")
