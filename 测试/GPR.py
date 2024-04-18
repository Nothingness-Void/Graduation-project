# 导入所需的 Python 包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pickle

# 读取数据
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵
featere_cols = ['MolWt1', 'logP1', 'TPSA1', #'n_h_donor1', 'n_h_acceptor1', 'total_charge1', 'bond_count1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 'LabuteASA1',
                'MolWt2', 'logP2', 'TPSA2', #'n_h_donor2', 'n_h_acceptor2', 'total_charge2', 'bond_count2',
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 'LabuteASA2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']

# 定义指纹特征矩阵
fingerprints = ['AvalonFP1', 'AvalonFP2', 'TopologicalFP1', 'TopologicalFP2', 'MorganFP1', 'MorganFP2']

# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[featere_cols], 
# data[fingerprints]
], axis=1)

# 定义目标参数
y = data['χ-result'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据X
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# 定义模型和参数空间
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2))  
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)  
param_space = {'alpha': Real(1e-6, 1e-2, prior='log-uniform')}  

# 创建贝叶斯搜索对象
optimizer = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=50, cv=5, 
                          scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1)

# 进行贝叶斯优化
optimizer.fit(X_train, y_train)

# 获取最优的参数和分数
best_param = optimizer.best_params_
best_score = optimizer.best_score_

# 在测试集上评估最优的模型
y_pred = optimizer.predict(X_test)

# 计算 R2 分数和 MSE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 打印最优模型的评估结果
print(f"最优模型: {model.__class__.__name__}")
print(f"最优参数: {best_param}")
print(f"Best Score: {best_score}")
print(f"R2: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# 保存最优的模型
with open('fingerprint_model.pkl', 'wb') as f:
    pickle.dump(optimizer.best_estimator_, f)
print("最优模型已储存为 fingerprint_model.pkl")