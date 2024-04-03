# author: code_king
# time: 2023/5/19 10:24
# file: develop_model1.py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def get_x_y(x_filename, y_filename, col):
    """
    :param x_filename:
    :param y_filename:
    :param col: 取出y的第几列
    :return:
    """
    x_data = pd.read_excel(io=x_filename, sheet_name=0, header=None)
    y_data = pd.read_excel(io=y_filename, sheet_name=0, header=None)
    # excel转np
    develop_x = x_data.to_numpy()
    # 这个4是第4列
    develop_y = y_data.to_numpy()[2:, col]
    merged_data = np.c_[develop_x, develop_y]
    # 记录需要删除的行
    del_rows = np.where(merged_data == "-")
    pure_merged_data = np.delete(merged_data, del_rows, axis=0)
    develop_x, develop_y = pure_merged_data[:, :-1], pure_merged_data[:, -1]
    return develop_x, develop_y


def pred_model(develop_x, develop_y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(develop_x, develop_y, test_size=0.2, random_state=42)
    # 创建 SVM 回归模型
    svm_regressor = SVR(kernel='rbf', C=1000)
    # 训练模型
    svm_regressor.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = svm_regressor.predict(X_test)
    # 打印预测结果
    print("预测结果:", y_pred)
    # 计算均方误差
    # mse = mean_squared_error(y_test, y_pred)
    # print("均方误差:", mse)
    # 计算 R 方
    r2 = r2_score(y_test, y_pred)
    print("R 方:", r2)


# 读取excel数据
develop_x, develop_y = get_x_y("./original_data/x.xlsx", "./original_data/fingerprint.xlsx", 5)
pred_model(develop_x, develop_y)
