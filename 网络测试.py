from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import urllib.parse

# 定义函数，将名称转换为SMILES
def CIRconvert(ids):
    try:
        encoded_component = urllib.parse.quote(ids)
        url = f'http://cactus.nci.nih.gov/chemical/structure/{encoded_component}/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'

print(CIRconvert('tetrahydrofuran'))