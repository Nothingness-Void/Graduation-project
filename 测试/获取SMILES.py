from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import urllib.parse
import time
import random

# 定义函数，将名称转换为SMILES
def CIRconvert(ids):
    try:
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
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_component}/property/CanonicalSMILES/TXT'
        
        # 在发送请求之前等待一段随机的时间
        time.sleep(random.uniform(0.5, 2.0))
        
        ans = urlopen(url).read().decode('utf8').strip()
        print(ans)
        return ans
    except Exception as e:
        print(str(e))
        return str(e)

# 读取excel文件
df = pd.read_excel('Huggins.xlsx')

# 创建一个线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 将名称转换为SMILES
    df['SMILES 1'] = list(tqdm(executor.map(CIRconvert, df['Compound 1']), total=df.shape[0]))
    df['SMILES 2'] = list(tqdm(executor.map(CIRconvert, df['Compound 2']), total=df.shape[0]))

# 新存储为csv文件
df.to_csv('SMILES.csv', index=False,encoding='utf_8_sig')