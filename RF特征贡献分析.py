"""

随机森林特征重要性分析脚本
从训练好的模型中提取 feature_importances_，绘制排序后的特征重要性柱状图

"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from feature_config import SELECTED_FEATURE_COLS

# ========== 配置 ==========
MODEL_PATH = "results/fingerprint_model.pkl"        # 模型文件路径
OUTPUT_PATH = "results/RF_feature_importance.png"   # 输出图片路径

# 加载模型
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"已加载模型: {MODEL_PATH} ({model.__class__.__name__})")

# 定义特征矩阵
feature_cols = np.array(SELECTED_FEATURE_COLS)

# 获取特征重要性
importances = model.feature_importances_

# 获取特征重要性排序的索引
sorted_idx = np.argsort(importances)

# 打印排序后的特征重要性
print("\n============ 特征重要性排序 ============")
for i in reversed(sorted_idx):
    print(f"  {feature_cols[i]:35s}  {importances[i]:.4f}")

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_cols[sorted_idx])
plt.xlabel('Importance')
plt.title(f'Feature Importances ({model.__class__.__name__})')
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\n特征重要性图已保存至: {OUTPUT_PATH}")
plt.show()
