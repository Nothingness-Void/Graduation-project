import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from feature_config import SELECTED_FEATURE_COLS

# 加载最优的随机森林模型 (假设已保存为 fingerprint_model.pkl)
with open("fingerprint_model.pkl", "rb") as f:
    model = pickle.load(f)

# 定义特征矩阵
feature_cols = np.array(SELECTED_FEATURE_COLS)

# 获取特征重要性
importances = model.feature_importances_

# 获取特征重要性排序的索引
sorted_idx = np.argsort(importances)

# 绘制特征重要性图
plt.figure(figsize=(12, 7))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_cols[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
