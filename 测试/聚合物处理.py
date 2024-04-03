import urllib.parse
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# 定义函数，将名称转换为SMILES
def CIRconvert(ids):
    # 检查化学名称是否包含聚合物
    if 'poly(' in ids.lower() and ids.endswith(')'):
        # 找到"poly("和")"的位置
        start = ids.index('poly(') + 5
        end = ids.index(')', start)

        # 只取出括号内的部分
        ids = ids[start:end]

    elif 'poly' in ids.lower():
        # 如果没有括号，去掉'poly'部分
        ids = ids[4:]

    encoded_component = urllib.parse.quote(ids)
    return encoded_component

# 读取excel文件
df = pd.read_excel('Huggins.xlsx')

# 创建一个线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 将名称转换为SMILES
    df['SMILES 1'] = list(tqdm(executor.map(CIRconvert, df['Compound 1']), total=df.shape[0]))
    df['SMILES 2'] = list(tqdm(executor.map(CIRconvert, df['Compound 2']), total=df.shape[0]))

# 新存储为csv文件
df.to_csv('简化.csv', index=False,encoding='utf_8_sig')
