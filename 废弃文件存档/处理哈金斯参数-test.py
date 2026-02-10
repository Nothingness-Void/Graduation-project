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
        part = part.replace('T', '').strip()  # 去除原始部分的T和空格
        try:
            part_result = float(part) if part else 0.0  # 尝试转换为浮点数
            if 'T' in original_part:
                result += sign * part_result / T  # 如果部分包含T，那么将其除以T并加到结果上
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

# 裂变数据
def split_row(row, index):
    chi = row['χ']
    T = row['Measured at T (K)']
    # 如果χ列没有包含T，那么不需要裂变
    if 'T' not in chi:
        chi, error = process_chi(chi, process_T(T))
        row['χ'] = chi
        row['Measured at T (K)'] = process_T(T)
        if error:
            row['Type'] = error
        else:
            row['Type'] = '原始数据'
        return pd.DataFrame([row], index=[index])
    # 否则，需要裂变
    rows = []
    if pd.isnull(T):  # 如果T是空值
        temperatures = [273.15, 323.15, 348.15, 373.15]  # 0, 50, 75, 100摄氏度对应的开尔文温度
        types = ['0摄氏度', '50摄氏度', '75摄氏度', '100摄氏度']
    elif '-' in T:  # 如果T是浮动值
        T1, T2 = map(float, T.split('-'))
        temperatures = [T1, T1 + (T2 - T1) * 0.25, T1 + (T2 - T1) * 0.75, T2]
        types = ['最小值', '下四分位', '上四分位', '最大值']
    else:  # 如果T是固定值
        temperatures = [float(T)]
        types = ['计算数据']

    for temperature, type in zip(temperatures, types):
        new_row = row.copy()
        new_row['χ'], error = process_chi(chi, temperature)
        new_row['Measured at T (K)'] = temperature
        new_row['Type'] = type
        rows.append(new_row)
    return pd.DataFrame(rows, index=[index + i / 1000 for i in range(1, len(rows) + 1)])

# 使用裂变函数处理数据
frames = []
for index, row in df.iterrows():
    frames.append(split_row(row, index))
df = pd.concat(frames).sort_index()

# 保存处理后的数据
df.to_excel('processed_and_split_Smiles_test.xlsx', index=False)