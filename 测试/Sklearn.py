# 导入所需的 Python 包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

target_name = 'χ-result'  # 哈金斯参数的表头

# 读取数据文件
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵
X = data[['Similarity', 'MolWt1', 'logP1', 'TPSA1', 'MolWt2', 'logP2', 'TPSA2']].values

print(X.shape)

# 定义目标参数
y = data[target_name].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据X
scaler_x = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 标准化数据y
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

########################################### 分割线 ####################################


# 定义机器学习算法列表
models = [  # LinearRegression(),
    Ridge(),
    Lasso(),
    SVR(kernel='rbf'),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    #MLPRegressor()
]

# 定义网格搜索参数列表
params = [
    {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000], 'max_iter': [5000, 10000, 20000]},#Ridge
    {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000], 'max_iter': [5000, 10000, 20000]},#Lasso
    {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 1, 10, 100, 1000, 1200, 1500],
     'gamma': [0.01, 0.05, 0.1, 1, 10, 100, 1000], 'max_iter': [1000, 5000, 10000]},#SVR
    {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [None, 3, 5, 10, 20],
     'max_features': [None, 'sqrt', 'log2']},#RandomForestRegressor
    {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [None, 3, 5, 10, 20],
     'max_features': [None, 'sqrt', 'log2']},#GradientBoostingRegressor
    #{'hidden_layer_sizes': [(10,), (20,), (50,), (100,)], 'activation': ['relu', 'tanh', 'logistic'], 'solver': ['adam', 'sgd', 'lbfgs'], 'max_iter': [1000]}
]

# 创建结果 DataFrame
results = pd.DataFrame(columns=['Model', 'Best parameter', 'Best score', 'R2', 'MAE', 'RMSE'])

result_models = []  # 用于存储每个模型的最优模型
for model, param in tqdm(zip(models, params), total=len(models), desc='正在建模'):
    # 创建网格搜索对象
    grid = GridSearchCV(model, param, cv=5, scoring='neg_mean_squared_error')
    # 在训练集上进行网格搜索
    grid.fit(X_train, y_train)

    # 获取最优的参数和分数
    best_param = grid.best_params_
    best_score = grid.best_score_

    # 反标准化
    y_pred_origin = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_test_origin = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # 在测试集上评估最优的模型
    y_pred = grid.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # 打印模型的名称，最优参数，最优分数，以及在测试集上的 R2 分数和 MSE
    print(f"当前模型: {model.__class__.__name__}")
    print(f"最优参数: {best_param}")
    print(f"Best Score: {best_score}")
    print(f"R2: {r2}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print("\n")

    # 将结果添加到结果 DataFrame
    new_row = pd.DataFrame(
        {'Model': [model.__class__.__name__], 'Best parameter': [best_param], 'Best score': [best_score], 'R2': [r2],
         'MAE': [mae], 'MaPE': [mape]})
    results = pd.concat([results, new_row], ignore_index=True)
    # 将最优模型添加到列表
    result_models.append(grid.best_estimator_)

results['R2'] = results['R2'].astype(float)
results['MAE'] = results['MAE'].astype(float)
results['MaPE'] = results['MaPE'].astype(float)

# 选择最优的模型
best_model_index = results['R2'].idxmax()
result_model = result_models[best_model_index]

# 保存最优的模型
with open('fingerprint_model.pkl', 'wb') as f:
    pickle.dump(result_model, f)
print("最优模型已储存为 fingerprint_model.pkl")

# 输出最优的模型的参数和结果
print('最优模型:', result_model.__class__.__name__)
print('最优参数：', results['Best parameter'][best_model_index])
print('R²:', results['R2'].max())
print('MAE(平均绝对误差):', results['MAE'].min())
print('MAPE(平均绝对百分比误差):', results['MaPE'].min())

