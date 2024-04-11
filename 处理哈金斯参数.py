"""

处理原始哈金斯参数数据中夹杂的浮动值和温度并计算出结果输出

"""
import pandas as pd
import numpy as np
import re

# 读取csv文件
df = pd.read_csv('Smiles.csv')

# 对χ列进行处理

import re

def process_chi(chi, T):
    chi = chi.replace('\xa0', '')  # 删除非断空格字符
    chi = chi.replace('(', '').replace(')', '')  # 删除括号
    parts = re.split(r'(\+|-)', chi)  # 使用正则表达式分割字符串，同时保留分割符号
    result = 0.0
    error = ''  # 用于记录错误的变量
    sign = 1  # 用于记录当前符号的变量
    for part in parts:
        original_part = part
        if part in ['+', '-']:  # 如果部分是分割符号
            sign = 1 if part == '+' else -1  # 更新符号
            continue
        if '±' in part:
            part = part.split('±')[0]  # 只取原始值
        part = part.replace('T', '').strip()  # 去除T和空格
        try:
            part_result = float(part) if part else 0.0  # 尝试转换为浮点数
            if 'T' in original_part:
                result += sign * part_result / T  # 如果部分包含T，那么将其乘以T并加到结果上
            else:
                result += sign * part_result  # 否则，将其加到结果上
        except ValueError:  # 如果转换失败，记录错误信息
            error += f'Could not convert "{part}" to float.\n'
    return result, error  # 返回结果和错误信息
    
# 对Measured at T (K)列进行处理
def process_T(T):
    if pd.isnull(T):
        return 298.15
    elif '-' in T:
        numbers = list(map(float, T.split('-')))
        return sum(numbers) / len(numbers)
    else:
        return float(T)

# 应用函数处理数据
df['χ'] = df.apply(lambda row: process_chi(row['χ'], process_T(row['Measured at T (K)']))[0], axis=1)
df['Measured at T (K)'] = df['Measured at T (K)'].apply(process_T)

# 保存处理后的数据
df.to_csv('processed_Smiles.csv', index=False)
