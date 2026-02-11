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
data = pd.read_excel('data/huggins_preprocessed.xlsx')

# 初始化结果列表
results = []

# 使用 tqdm 函数来显示进度条
for i,row in tqdm(data.iterrows(), total=len(data), desc="处理中……"):

    # 获取 SMILES 字符串
    smiles1, smiles2 = row[smiles1_name], row[smiles2_name]

    # 从 SMILES 字符串创建分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:
        mol1 = Chem.AddHs(mol1)
        mol2 = Chem.AddHs(mol2)
        
        #计算分子指纹

        # 计算 Avalon 指纹
        Avalon_fingerprint1 = pyAvalonTools.GetAvalonFP(mol1)
        Avalon_fingerprint2 = pyAvalonTools.GetAvalonFP(mol2)

        # 计算 Morgan 指纹
        Morgan_fingerprint1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        Morgan_fingerprint2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        # 计算拓扑结构指纹
        Topological_fingerprint1 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol1)
        Topological_fingerprint2 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol2)


        # 计算结构相似度
        Morgan_Similar = FingerprintSimilarity(Morgan_fingerprint1, Morgan_fingerprint2) # 利用Tanimoto系数计算相似度

        # 计算指纹相似度
        Avalon_Similar = FingerprintSimilarity(Avalon_fingerprint1, Avalon_fingerprint2)  # 利用Avalon指纹计算相似度

        # 计算拓补结构指纹相似度
        Topological_Similar = FingerprintSimilarity(Topological_fingerprint1, Topological_fingerprint2) # 利用Tanimoto系数计算相似度



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
        inertial_shape_factor1 = rdMolDescriptors.CalcInertialShapeFactor(mol1)
        inertial_shape_factor2 = rdMolDescriptors.CalcInertialShapeFactor(mol2)
        # 计算分子的主惯性比
        mol1_npr1 = rdMolDescriptors.CalcNPR1(mol1)
        mol1_npr2 = rdMolDescriptors.CalcNPR2(mol1)
        mol2_npr1 = rdMolDescriptors.CalcNPR1(mol2)
        mol2_npr2 = rdMolDescriptors.CalcNPR2(mol2)
        # 计算最大绝对偏电荷（极性描述符）
        MaxAbsPartialCharge1 = Descriptors.MaxAbsPartialCharge(mol1)
        MaxAbsPartialCharge2 = Descriptors.MaxAbsPartialCharge(mol2)
        # 计算可旋转键的数量
        rotatable_bonds1 = rdMolDescriptors.CalcNumRotatableBonds(mol1)
        rotatable_bonds2 = rdMolDescriptors.CalcNumRotatableBonds(mol2)
        # 计算分子的球形度指数
        CalcSpherocityIndex1 = rdMolDescriptors.CalcSpherocityIndex(mol1)
        CalcSpherocityIndex2 = rdMolDescriptors.CalcSpherocityIndex(mol2)
        # 计算分子的回旋半径    
        CalcRadiusOfGyration1 = rdMolDescriptors.CalcRadiusOfGyration(mol1)
        CalcRadiusOfGyration2 = rdMolDescriptors.CalcRadiusOfGyration(mol2)



        # QSPR描述符
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

        # 计算分子的Labute 约化表面积
        LabuteASA1 = rdMolDescriptors.CalcLabuteASA(mol1)
        LabuteASA2 = rdMolDescriptors.CalcLabuteASA(mol2)

        #计算杂原子数量
        atom_count1 = Descriptors.NumHeteroatoms(mol1)
        atom_count2 = Descriptors.NumHeteroatoms(mol2)

        # ===== 新增交互特征 =====
        # 热力学驱动力：疏水性差异（χ ∝ (δ₁-δ₂)²，LogP 差异近似）
        Delta_LogP = abs(logp1 - logp2)
        # 极性不匹配度
        Delta_TPSA = abs(tpsa1 - tpsa2)
        # 氢键匹配性：交叉项（供体1×受体2 + 受体1×供体2）
        HB_Match = n_h_donor1 * n_h_acceptor2 + n_h_acceptor1 * n_h_donor2
        # 分子体积差异（摩尔折射率 MolMR，与分子体积强相关）
        MolMR1 = Descriptors.MolMR(mol1)
        MolMR2 = Descriptors.MolMR(mol2)
        Delta_MolMR = abs(MolMR1 - MolMR2)
        # 分子柔性（sp3 碳比例）
        CSP3_1 = Descriptors.FractionCSP3(mol1)
        CSP3_2 = Descriptors.FractionCSP3(mol2)
        # 温度物理项（χ = A + B/T，1/T 更符合物理规律）
        Inv_T = 1000.0 / row['Measured at T (K)']

        # 将结果添加到结果列表中
        results.append({
            'MolWt1': mol_wt1,
            'logP1': logp1,
            'TPSA1': tpsa1,
            'asphericity1': asphericity1,
            'eccentricity1': eccentricity1,
            'inertial_shape_factor1': inertial_shape_factor1,
            'mol1_npr1': mol1_npr1,
            'mol1_npr2': mol1_npr2,
            'MaxAbsPartialCharge1': MaxAbsPartialCharge1,
            'LabuteASA1': LabuteASA1,
            'CalcSpherocityIndex1': CalcSpherocityIndex1,
            'CalcRadiusOfGyration1': CalcRadiusOfGyration1,
            'MolWt2': mol_wt2,
            'logP2': logp2,
            'TPSA2': tpsa2,
            'asphericity2': asphericity2,
            'eccentricity2': eccentricity2,
            'inertial_shape_factor2': inertial_shape_factor2,
            'mol2_npr1': mol2_npr1,
            'mol2_npr2': mol2_npr2,
            'MaxAbsPartialCharge2': MaxAbsPartialCharge2,
            'LabuteASA2': LabuteASA2,
            'CalcSpherocityIndex2': CalcSpherocityIndex2,
            'CalcRadiusOfGyration2': CalcRadiusOfGyration2,
            'Avalon Similarity': Avalon_Similar,
            'Morgan Similarity': Morgan_Similar,
            'Topological Similarity': Topological_Similar,
            # ===== 交互特征 =====
            'Delta_LogP': Delta_LogP,
            'Delta_TPSA': Delta_TPSA,
            'HB_Match': HB_Match,
            'Delta_MolMR': Delta_MolMR,
            'CSP3_1': CSP3_1,
            'CSP3_2': CSP3_2,
            'Inv_T': Inv_T,
            'χ-result': row[target_name],
        })
    else:
        print(f"Unable to parse SMILES string: {smiles1} or {smiles2}")

# 将结果列表转换为 DataFrame
result = pd.DataFrame(results)

# 保存结果到 Excel 文件
result.to_excel('data/molecular_features.xlsx', index=False)
