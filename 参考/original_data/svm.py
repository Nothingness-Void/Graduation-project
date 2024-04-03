import pandas as pd
import numpy as np

from sklearn import svm

from matplotlib import font_manager
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from numpy import mat
import warnings
warnings.filterwarnings("ignore")


#读取y值
    # 从 Excel 文件中读取数据
df = pd.read_excel('fingerprint.xlsx', header=None, skiprows=2, usecols='E')

# 提取有效数据
valid_data = df.iloc[2:128, 0].values

# 创建二维数组并初始化为0
temp = np.zeros(shape=(124,1),dtype=float)

# 将有效数据填充到指纹数组中
for i, data in enumerate(valid_data):
    temp[i] = pd.to_numeric(data, errors='coerce', downcast='integer')

print(temp)


#读取x值
   # 读取Excel文件
df = pd.read_excel('x.xlsx', header=None)

# 创建一个空字典来存储每一列的数组
fingerprint = {}

# 遍历每一列，并将其存储为一个数组
for column in df.columns:
    column_data = df[column].values.tolist()
    fingerprint[column] = column_data

# 输出每一列的数组
for column, data in fingerprint.items():
    print(f'Column {column}: {data}')


#svm建模
# 创建模型
models = {}

# 遍历字典的值
for key, value in enumerate(fingerprint.items()):
    # 获取对应的目标变量 y
    y = np.array(temp[key]).reshape(-1, 1)
    x = np.array(value).reshape(-1, 1)
    # 创建 SVR 模型并进行拟合
    model = svm.SVR(kernel='linear')
    model.fit(x, y)
    
    # 将模型存储到字典中
    models[i] = model

print(models)