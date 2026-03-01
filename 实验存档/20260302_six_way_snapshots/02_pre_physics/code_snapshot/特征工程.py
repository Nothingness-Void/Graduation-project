# -*- coding: utf-8 -*-
"""
特征工程.py - 全量 RDKit 分子描述符提取 + 交互特征

使用 RDKit Descriptors.CalcMolDescriptors() 自动提取全部 ~210 个描述符，
对聚合物+溶剂两个分子分别计算后拼接，再补充交互特征和指纹相似度。

输入: data/merged_dataset.csv
输出: data/molecular_features.xlsx
"""
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.DataStructs import FingerprintSimilarity
from tqdm import tqdm

# 抑制 RDKit 警告 (对聚合物 SMILES [*] 替换后仍可能有少量警告)
RDLogger.logger().setLevel(RDLogger.ERROR)

# ========== 配置 ==========
INPUT_PATH = "data/merged_dataset.csv"
OUTPUT_PATH = "data/molecular_features.xlsx"

TARGET_COL = "chi"
SMILES1_COL = "Polymer_SMILES"
SMILES2_COL = "Solvent_SMILES"
TEMP_COL = "temperature"


# ========== SMILES 预处理 ==========
def sanitize_smiles(smi: str) -> str:
    """
    处理聚合物 SMILES 中的 [*] 连接点标记。
    策略: 用 [H] 替换 [*]，使 RDKit 能正常解析。
    """
    if not isinstance(smi, str):
        return ""
    # 替换常见的连接点标记
    smi = smi.replace("[*]", "[H]")
    smi = smi.replace("*", "")
    return smi.strip()


def safe_mol(smi: str):
    """安全地从 SMILES 创建分子对象，失败返回 None。"""
    smi = sanitize_smiles(smi)
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            pass
    return mol


# ========== 全量描述符计算 ==========
# 获取所有 RDKit 2D 描述符名称
ALL_DESC_NAMES = [x[0] for x in Descriptors._descList]


def calc_all_descriptors(mol) -> dict:
    """
    使用 CalcMolDescriptors 计算全部 ~210 个 RDKit 2D 描述符。
    如果某个描述符计算失败，返回 NaN。
    """
    if mol is None:
        return {name: np.nan for name in ALL_DESC_NAMES}

    try:
        desc_dict = Descriptors.CalcMolDescriptors(mol)
    except Exception:
        desc_dict = {}
        for name, func in Descriptors._descList:
            try:
                desc_dict[name] = func(mol)
            except Exception:
                desc_dict[name] = np.nan

    # 确保所有值都是数值型（部分描述符可能返回非数值）
    result = {}
    for name in ALL_DESC_NAMES:
        val = desc_dict.get(name, np.nan)
        if val is None or (isinstance(val, float) and np.isinf(val)):
            result[name] = np.nan
        else:
            try:
                result[name] = float(val)
            except (TypeError, ValueError):
                result[name] = np.nan
    return result


# ========== 指纹相似度 ==========
def calc_fingerprint_similarities(mol1, mol2) -> dict:
    """
    计算三种分子指纹之间的 Tanimoto 相似度。
    """
    result = {
        "Avalon_Similarity": np.nan,
        "Morgan_Similarity": np.nan,
        "Topological_Similarity": np.nan,
    }
    if mol1 is None or mol2 is None:
        return result

    try:
        fp1 = pyAvalonTools.GetAvalonFP(mol1)
        fp2 = pyAvalonTools.GetAvalonFP(mol2)
        result["Avalon_Similarity"] = FingerprintSimilarity(fp1, fp2)
    except Exception:
        pass

    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        result["Morgan_Similarity"] = FingerprintSimilarity(fp1, fp2)
    except Exception:
        pass

    try:
        fp1 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol1)
        fp2 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol2)
        result["Topological_Similarity"] = FingerprintSimilarity(fp1, fp2)
    except Exception:
        pass

    return result


# ========== 交互特征 ==========
def calc_interaction_features(desc1: dict, desc2: dict, temperature: float) -> dict:
    """
    计算基于物理化学原理的交互特征:
    差值 (|polymer - solvent|) 和比值特征。
    """
    interaction = {}

    # 差值特征 (有物理意义的配对)
    diff_pairs = [
        ("MolLogP", "Delta_LogP"),
        ("TPSA", "Delta_TPSA"),
        ("MolMR", "Delta_MolMR"),
        ("ExactMolWt", "Delta_MolWt"),
        ("MaxAbsPartialCharge", "Delta_MaxAbsCharge"),
        ("HeavyAtomMolWt", "Delta_HeavyAtomMolWt"),
        ("NumHDonors", "Delta_HBD"),
        ("NumHAcceptors", "Delta_HBA"),
        ("FractionCSP3", "Delta_CSP3"),
        ("LabuteASA", "Delta_LabuteASA"),
        ("NumRotatableBonds", "Delta_RotBonds"),
        ("NumAromaticRings", "Delta_AromaticRings"),
    ]
    for d_name, feat_name in diff_pairs:
        v1 = desc1.get(d_name, np.nan)
        v2 = desc2.get(d_name, np.nan)
        try:
            interaction[feat_name] = abs(float(v1) - float(v2))
        except (TypeError, ValueError):
            interaction[feat_name] = np.nan

    # 氢键匹配性: 供体1×受体2 + 受体1×供体2
    try:
        hbd1 = float(desc1.get("NumHDonors", 0))
        hba1 = float(desc1.get("NumHAcceptors", 0))
        hbd2 = float(desc2.get("NumHDonors", 0))
        hba2 = float(desc2.get("NumHAcceptors", 0))
        interaction["HB_Match"] = hbd1 * hba2 + hba1 * hbd2
    except (TypeError, ValueError):
        interaction["HB_Match"] = np.nan

    # 温度物理项 (chi = A + B/T)
    try:
        interaction["Inv_T"] = 1000.0 / float(temperature)
    except (TypeError, ValueError, ZeroDivisionError):
        interaction["Inv_T"] = np.nan

    return interaction


# ========== 主流程 ==========
def main():
    print("加载数据...")
    data = pd.read_csv(INPUT_PATH)
    print(f"  输入: {data.shape[0]} 行, 列: {list(data.columns)}")
    print(f"  RDKit 描述符数: {len(ALL_DESC_NAMES)}")
    print(f"  预计输出特征数: ~{len(ALL_DESC_NAMES) * 2 + 3 + 15} 维")

    results = []
    failed = 0

    for i, row in tqdm(data.iterrows(), total=len(data), desc="计算描述符"):
        mol1 = safe_mol(str(row[SMILES1_COL]))
        mol2 = safe_mol(str(row[SMILES2_COL]))

        if mol1 is None and mol2 is None:
            failed += 1
            continue

        # 1) 全量描述符 (分别给聚合物和溶剂加后缀)
        desc1 = calc_all_descriptors(mol1)
        desc2 = calc_all_descriptors(mol2)

        feat_row = {}
        for name, val in desc1.items():
            feat_row[f"{name}_1"] = val
        for name, val in desc2.items():
            feat_row[f"{name}_2"] = val

        # 2) 指纹相似度
        fp_sims = calc_fingerprint_similarities(mol1, mol2)
        feat_row.update(fp_sims)

        # 3) 交互特征
        interactions = calc_interaction_features(desc1, desc2, row.get(TEMP_COL, np.nan))
        feat_row.update(interactions)

        # 4) 目标值
        feat_row["chi_result"] = row[TARGET_COL]

        results.append(feat_row)

    # 转为 DataFrame
    df_features = pd.DataFrame(results)

    # ========== 清洗 ==========
    original_cols = len(df_features.columns) - 1  # 减去 chi_result

    # 1) 删除全为 NaN 的列
    threshold = 0.5  # 超过 50% 缺失则删除该列
    nan_ratio = df_features.drop(columns=["chi_result"]).isnull().mean()
    cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
    if cols_to_drop:
        df_features = df_features.drop(columns=cols_to_drop)
        print(f"  删除高缺失率列 (>{threshold*100:.0f}%): {len(cols_to_drop)} 列")

    # 2) 删除方差为 0 的列 (常量列)
    feature_cols = [c for c in df_features.columns if c != "chi_result"]
    const_cols = []
    for c in feature_cols:
        if df_features[c].dropna().nunique() <= 1:
            const_cols.append(c)
    if const_cols:
        df_features = df_features.drop(columns=const_cols)
        print(f"  删除常量列: {len(const_cols)} 列")

    # 3) 填充剩余 NaN 为列中位数
    feature_cols = [c for c in df_features.columns if c != "chi_result"]
    nan_before = df_features[feature_cols].isnull().sum().sum()
    for c in feature_cols:
        if df_features[c].isnull().any():
            df_features[c] = df_features[c].fillna(df_features[c].median())
    nan_after = df_features[feature_cols].isnull().sum().sum()
    if nan_before > 0:
        print(f"  NaN 中位数填充: {nan_before} -> {nan_after}")

    final_feature_count = len(df_features.columns) - 1
    print(f"\n{'='*60}")
    print(f"特征工程完成")
    print(f"{'='*60}")
    print(f"  样本数: {len(df_features)} (失败: {failed})")
    print(f"  原始特征: {original_cols} -> 清洗后特征: {final_feature_count}")
    print(f"  包含: {len(ALL_DESC_NAMES)}×2 描述符 + 3 相似度 + 交互特征")
    print(f"  目标列: chi_result")

    # 保存
    df_features.to_excel(OUTPUT_PATH, index=False)
    print(f"\n已保存至: {OUTPUT_PATH}")

    # 打印特征类别统计
    poly_feats = [c for c in df_features.columns if c.endswith("_1")]
    solv_feats = [c for c in df_features.columns if c.endswith("_2")]
    sim_feats = [c for c in df_features.columns if "Similarity" in c]
    inter_feats = [c for c in df_features.columns if c.startswith("Delta_") or c in ("HB_Match", "Inv_T")]
    print(f"\n  聚合物描述符 (_1): {len(poly_feats)}")
    print(f"  溶剂描述符 (_2): {len(solv_feats)}")
    print(f"  指纹相似度: {len(sim_feats)}")
    print(f"  交互特征: {len(inter_feats)}")


if __name__ == "__main__":
    main()
