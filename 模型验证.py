import pandas as pd
import pickle
import numpy as np

# 读取数据文件
data = pd.read_excel('fingerprint.xlsx')
data = data.iloc[1:]  # 去掉第一行
data['分子指纹'] = data['分子指纹'].astype(str)  # 将分子指纹转换为字符串
X = np.array([list(map(int, list(x))) for x in data['分子指纹']])

# 加载模型
with open('fingerprint_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 对分子指纹进行预测
y_pred = model.predict(X)

# 将预测结果保存到result.csv文件中
result = pd.DataFrame({'预测结果': y_pred})
result.to_excel('模型验证.xlsx', index=False,encoding='utf_8_sig')
