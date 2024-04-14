import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from tqdm import tqdm

target_name = 'χ-result'
smiles1_name = 'SMILES 1'
smiles2_name = 'SMILES 2'

# 读取输入数据
data = pd.read_excel('Smiles-处理后.xlsx')

# 初始化结果列表
results = []

# 使用 tqdm 函数来显示进度条
for smiles1, smiles2 in tqdm(zip(data[smiles1_name], data[smiles2_name]), total=len(data), desc="处理中……"):

    # 从 SMILES 字符串创建分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:
        # 计算 Avalon 指纹
        fingerprint1 = pyAvalonTools.GetAvalonFP(mol1)
        fingerprint2 = pyAvalonTools.GetAvalonFP(mol2)

        # 计算指纹相似度
        F_Similar = FingerprintSimilarity(fingerprint1, fingerprint2)

        # 计算物理化学性质
        mol_wt1 = rdMolDescriptors.CalcExactMolWt(mol1)  # 分子量
        mol_wt2 = rdMolDescriptors.CalcExactMolWt(mol2)
        logp1 = Descriptors.MolLogP(mol1)  # logP
        logp2 = Descriptors.MolLogP(mol2)
        tpsa1 = rdMolDescriptors.CalcTPSA(mol1)  # TPSA
        tpsa2 = rdMolDescriptors.CalcTPSA(mol2)

        # 将结果添加到结果列表中
        results.append({
            'AvalonFP1': fingerprint1.ToBitString(),
            'MolWt1': mol_wt1,
            'logP1': logp1,
            'TPSA1': tpsa1,
            'AvalonFP2': fingerprint2.ToBitString(),
            'MolWt2': mol_wt2,
            'logP2': logp2,
            'TPSA2': tpsa2,
            'Similarity': F_Similar,
            'χ-result': data.loc[data[smiles1_name] == smiles1, target_name].values[0]
        })
    else:
        print(f"Unable to parse SMILES string: {smiles1} or {smiles2}")

# 将结果列表转换为 DataFrame
result = pd.DataFrame(results)

# 保存结果到 Excel 文件
result.to_excel('计算结果.xlsx', index=False)
