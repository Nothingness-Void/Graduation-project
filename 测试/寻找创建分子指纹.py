from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np


# 创建一个分子对象
mol = Chem.MolFromSmiles('CCO')

# 生成ECFPs（默认的半径为2，也就是ECFP4）
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

# 将指纹转换为位数组
fp_arr = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, fp_arr)

print(fp_arr)
