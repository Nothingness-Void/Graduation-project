"""

通过化合物名称获取SMILES字符串


"""

from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import urllib.parse
import time
import random
import requests

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
    
        # 编码化合物名称
        encoded_component = urllib.parse.quote(ids)

        # 一次查询-PubChem查询
        pubchem_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_component}/property/CanonicalSMILES/TXT'
        # 在发送请求之前等待一段随机的时间
        time.sleep(random.uniform(0.5, 2.0))
        response = requests.get(pubchem_url)
        if response.status_code == 200:
            print("查询成功", response.text.strip())
            ans = urlopen(pubchem_url).read().decode('utf8').strip()
            return ans
        else:
            print("一次查询返回异常, 尝试二次查询...")
            # 二次查询
            chemspider_url = f'http://cactus.nci.nih.gov/chemical/structure/{encoded_component}/smiles'
            # 在发送请求之前等待一段随机的时间
            time.sleep(random.uniform(0.5, 2.0))
            response = requests.get(chemspider_url)
            if response.status_code == 200:
                print("二次查询成功:", response.text.strip())
                ans = urlopen(chemspider_url).read().decode('utf8').strip()
                return ans
            else:
                print("二次查询失败:", response.status_code)
                ans = urlopen(chemspider_url).read().decode('utf8').strip()
                return ans
    except Exception as e:
        print("Error:", str(e))
        return "Error"

# 读取excel文件
df = pd.read_excel('Huggins.xlsx')

# 创建一个线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 将名称转换为SMILES
    df['SMILES 1'] = list(tqdm(executor.map(CIRconvert, df['Compound 1']), total=df.shape[0]))
    #df['SMILES 2'] = list(tqdm(executor.map(CIRconvert, df['Compound 2']), total=df.shape[0]))

# 新存储为csv文件
df.to_csv('data/smiles_raw.csv', index=False,encoding='utf_8_sig')
