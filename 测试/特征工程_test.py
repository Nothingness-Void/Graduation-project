import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from tqdm import tqdm
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem

target_name = 'χ'
smiles1_name = 'SMILES 1'
smiles2_name = 'SMILES 2'

# 读取输入数据
data = pd.read_excel('processed_and_split_Smiles.xlsx')

# 初始化结果列表
results = []

# 使用 tqdm 函数来显示进度条
for smiles1, smiles2 in tqdm(zip(data[smiles1_name], data[smiles2_name]), total=len(data), desc="处理中……"):

    # 从 SMILES 字符串创建分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:

        # 计算 Avalon 指纹
        Avalon_fingerprint1 = pyAvalonTools.GetAvalonFP(mol1)
        Avalon_fingerprint2 = pyAvalonTools.GetAvalonFP(mol2)

        # 计算 Morgan 指纹
        Morgan_fingerprint1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        Morgan_fingerprint2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        # 计算结构相似度
        Morgan_Similar = DataStructs.TanimotoSimilarity(Morgan_fingerprint1, Morgan_fingerprint2) # 利用Tanimoto系数计算相似度

        # 计算指纹相似度
        Avalon_Similar = FingerprintSimilarity(Avalon_fingerprint1, Avalon_fingerprint2)  # 利用Avalon指纹计算相似度

        #创建分子3D结构
        AllChem.EmbedMolecule(mol1)
        AllChem.EmbedMolecule(mol2)
        #优化分子3D结构
        AllChem.MMFFOptimizeMolecule(mol1)
        AllChem.MMFFOptimizeMolecule(mol2)

        # 计算立体描述符
        # 计算分子的不对称性
        asphericity1 = rdMolDescriptors.CalcAsphericity(mol1)
        asphericity2 = rdMolDescriptors.CalcAsphericity(mol2)
        # 计算分子的偏心率
        eccentricity1 = rdMolDescriptors.CalcEccentricity(mol1)
        eccentricity2 = rdMolDescriptors.CalcEccentricity(mol2)
        # 计算分子的惯性形状因子
        inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(mol1)
        inertial_shape_factor2 = rdMolDescriptors.CalcInertialShapeFactor(mol2)
        # 计算分子的主惯性比
        mol1_npr1 = rdMolDescriptors.CalcNPR1(mol1)
        mol1_npr2 = rdMolDescriptors.CalcNPR2(mol1)
        mol2_npr1 = rdMolDescriptors.CalcNPR1(mol2)
        mol2_npr2 = rdMolDescriptors.CalcNPR2(mol2)

        # 计算物理化学性质
        # 分子量
        mol_wt1 = rdMolDescriptors.CalcExactMolWt(mol1)
        mol_wt2 = rdMolDescriptors.CalcExactMolWt(mol2)
        # 疏水性
        logp1 = Descriptors.MolLogP(mol1)
        logp2 = Descriptors.MolLogP(mol2)
        # 极性表面积
        tpsa1 = rdMolDescriptors.CalcTPSA(mol1)  
        tpsa2 = rdMolDescriptors.CalcTPSA(mol2)
        # 计算氢键供体和受体的数量
        n_h_donor1 = rdMolDescriptors.CalcNumHBD(mol1)
        n_h_donor2 = rdMolDescriptors.CalcNumHBD(mol2)
        n_h_acceptor1 = rdMolDescriptors.CalcNumHBA(mol1)
        n_h_acceptor2 = rdMolDescriptors.CalcNumHBA(mol2)
        # 计算分子总电荷
        total_charge1 = sum(atom.GetFormalCharge() for atom in mol1.GetAtoms())
        total_charge2 = sum(atom.GetFormalCharge() for atom in mol2.GetAtoms())
        # 计算分子键数量
        bond_types1 = set(bond.GetBondType() for bond in mol1.GetBonds())
        bond_count1 = len(bond_types1)
        bond_types2 = set(bond.GetBondType() for bond in mol2.GetBonds())
        bond_count2 = len(bond_types2)
        


        # 将结果添加到结果列表中
        results.append({
            'AvalonFP1': Avalon_fingerprint1.ToBitString(),
            'MorganFP1': Morgan_fingerprint1.ToBitString(),
            'MolWt1': mol_wt1,
            'logP1': logp1,
            'TPSA1': tpsa1,
            'n_h_donor1': n_h_donor1,
            'n_h_acceptor1': n_h_acceptor1,
            'total_charge1': total_charge1,
            'bond_count1': bond_count1,
            'AvalonFP2': Avalon_fingerprint2.ToBitString(),
            'MorganFP2': Morgan_fingerprint2.ToBitString(),
            'MolWt2': mol_wt2,
            'logP2': logp2,
            'TPSA2': tpsa2,
            'n_h_donor2': n_h_donor2,
            'n_h_acceptor2': n_h_acceptor2,
            'total_charge2': total_charge2,
            'bond_count2': bond_count2,
            'Similarity': Avalon_Similar,
            'Morgan Similarity': Morgan_Similar,
            'Measured at T (K)': data.loc[data[smiles1_name] == smiles1, 'Measured at T (K)'].values[0],
            'χ-result': data.loc[data[smiles1_name] == smiles1, target_name].values[0]
        })
    else:
        print(f"Unable to parse SMILES string: {smiles1} or {smiles2}")

# 将结果列表转换为 DataFrame
result = pd.DataFrame(results)

# 保存结果到 Excel 文件
result.to_excel('计算结果.xlsx', index=False)
