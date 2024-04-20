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
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import xgboost as xgb

target_name = 'χ-result'  # 哈金斯参数的表头

# 读取数据文件
data = pd.read_excel('计算结果.xlsx')

# 定义特征矩阵
featere_cols = ['MolWt1', 'logP1', 'TPSA1', #'n_h_donor1', 'n_h_acceptor1', 'total_charge1', 'bond_count1',
                'asphericity1', 'eccentricity1', 'inertial_shape_factor1', 'mol1_npr1', 'mol1_npr2', 'dipole1', 
                'LabuteASA1','CalcSpherocityIndex1','CalcRadiusOfGyration1','atom_count1',
                'MolWt2', 'logP2', 'TPSA2', #'n_h_donor2', 'n_h_acceptor2', 'total_charge2', 'bond_count2',
                'asphericity2', 'eccentricity2', 'inertial_shape_factor2', 'mol2_npr1', 'mol2_npr2', 'dipole2', 
                'LabuteASA2','CalcSpherocityIndex2','CalcRadiusOfGyration2','atom_count2',
                'Avalon Similarity', 'Morgan Similarity', 'Topological Similarity', 'Measured at T (K)']

# 定义指纹特征矩阵
fingerprints = ['AvalonFP1', 'AvalonFP2', 'TopologicalFP1', 'TopologicalFP2', 'MorganFP1', 'MorganFP2']


# 将编码后的指纹特征和数值特征合并
X = pd.concat([data[featere_cols], 
# data[fingerprints]
], axis=1)


# 定义目标参数
y = data[target_name].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据X
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)


# 标准化数据y
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()


########################################### 分割线 ####################################


# 定义模型和参数空间
models = [
    (Ridge(), {'alpha': Real(1e-3, 1e3, prior='log-uniform'), 'max_iter': Integer(1000, 20000)}),
    (Lasso(), {'alpha': Real(1e-3, 1e3, prior='log-uniform'), 'max_iter': Integer(1000, 20000)}),
    (SVR(), {'C': Real(1e-3, 1e3, prior='log-uniform'), 'kernel': Categorical(['linear', 'rbf', 'poly']),
             'gamma': Real(1e-3, 1e3, prior='log-uniform'), 'max_iter': Integer(50, 5000)}),
    (RandomForestRegressor(), {'n_estimators': Integer(10, 500), 'max_depth': Integer(3, 20),
                               'max_features': Categorical(['sqrt', 'log2'])}),
    (GradientBoostingRegressor(), {'n_estimators': Integer(10, 500), 'max_depth': Integer(3, 20),
                                   'max_features': Categorical(['sqrt', 'log2'])}),  
    # 添加XGBRegressor
    (xgb.XGBRegressor(), {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.5, 1, prior='uniform'),
        'colsample_bytree': Real(0.5, 1, prior='uniform'),
    }),  
    (MLPRegressor(), {'hidden_layer_sizes': Integer(1, 1000), 'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),}),
]

# 创建结果 DataFrame
results = pd.DataFrame(columns=['Model', 'Best parameter', 'Best score', 'R2', 'MAE', 'MaPE'])

result_models = []  # 用于存储每个模型的最优模型

for model, param_space in tqdm(models, total=len(models), desc='正在建模'):
    # 创建贝叶斯优化对象
    #grid = GridSearchCV(model, param, scoring='r2', cv=5, n_jobs=-1)
    grid = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=75,  # 迭代次数
        cv=5,  # 交叉验证折数
        scoring='neg_mean_squared_error',  # 评估指标
        n_jobs=-1,  # 使用所有 CPU 核心
    )
    
    # 在训练集上进行网格搜索
    grid.fit(X_train, y_train)

    # 获取最优的参数和分数
    best_param = grid.best_params_
    best_score = grid.best_score_

    # 在测试集上评估最优的模型
    y_pred = grid.predict(X_test)


    # 反标准化
    y_pred_origin = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_test_origin = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # 计算 R2 分数和 MSE
    r2 = r2_score(y_test_origin, y_pred_origin)
    mae = mean_absolute_error(y_test_origin, y_pred_origin)
    rmse = np.sqrt(mean_squared_error(y_test_origin, y_pred_origin))

    # 打印模型的名称，最优参数，最优分数，以及在测试集上的 R2 分数和 MSE
    print(f"当前模型: {model.__class__.__name__}")
    print(f"最优参数: {best_param}")
    print(f"Best Score: {best_score}")
    print(f"R2: {r2}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print("\n")

    # 将结果添加到结果 DataFrame
    new_row = pd.DataFrame(
        {'Model': [model.__class__.__name__], 'Best parameter': [best_param], 'Best score': [best_score], 'R2': [r2],
         'MAE': [mae], 'RMSE': [rmse]})
    results = pd.concat([results, new_row], ignore_index=True)
    # 将最优模型添加到列表
    result_models.append(grid.best_estimator_)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 打印模型的名称，最优参数，最优分数，以及在测试集上的 R2 分数和 MSE
    print(f"当前模型: {model.__class__.__name__}")
    print(f"最优参数: {best_param}")
    print(f"Best Score: {best_score}")
    print(f"R2: {r2}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print("\n")

    # 将结果添加到结果 DataFrame
    new_row = pd.DataFrame(
        {'Model': [model.__class__.__name__], 'Best parameter': [best_param], 'Best score': [best_score], 'R2': [r2],
         'MAE': [mae], 'RMSE': [rmse]})
    results = pd.concat([results, new_row], ignore_index=True)
    # 将最优模型添加到列表
    result_models.append(grid.best_estimator_)

results['R2'] = results['R2'].astype(float)
results['MAE'] = results['MAE'].astype(float)
results['RMSE'] = results['RMSE'].astype(float)

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
print('RMSE(均方根误差):', results['RMSE'].min())

