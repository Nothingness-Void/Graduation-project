from rdkit import Chem
from rdkit.Chem import Descriptors

# 创建一个分子
smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
molecule = Chem.MolFromSmiles(smiles)

# 计算分子描述符
mol_weight = Descriptors.MolWt(molecule)
logp = Descriptors.MolLogP(molecule)

print(f"Molecular Weight: {mol_weight}")
print(f"LogP: {logp}")
