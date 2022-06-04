# 用python实现，吴恩达机器学习慕课-实验一-中的多变量回归任务
# 2022年6月04日10:01:46

import numpy as np
import pandas as pd

import singleVariableRegression as sv


# 读取数据-相对于单变量回归需要进行一下，特征归一化
def importDataFromFile(path):
    data = pd.read_csv(path)

    cols = data.shape[1]  # 列数
    X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
    y = data.iloc[:, cols - 1:cols]  # X是所有行，最后一列

    # 将pandas读取进来的数据转换成numpy的矩阵形式, 下面的X，y，theta用来返回
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # 多变量回归需要进行特征归一化
    mu = np.mean(X, 0)
    std = np.std(X, 0)
    X = (X - mu) / std

    # 训练集前面插入一列作为截距项x_0
    X = np.c_[np.ones(X.shape[0]), X]

    # theta = np.matrix(np.array([0, 0]))
    theta = np.zeros((1, X.shape[1]))

    return X, y, theta, data, mu, std


# 预测-注意进行特征归一化
def predictProfit(area, roomNum, theta):
    price = theta[0, 0] + theta[0, 1] * area + theta[0, 2] * roomNum;
    return price


if __name__ == '__main__':
    # 指定数据路径
    path = 'ex1data2.txt'
    # 定义学习率和迭代次数-超参数
    alpha = 0.01
    iters = 10000

    # 读入数据, 在此X为m行n+1列，也就是m行2列，y是m行1列，theta是1行2列
    X, y, theta, data, mu, std = importDataFromFile(path)

    # 进行梯度下降，训练求解
    theta, cost = sv.gradientDescent(X, y, theta, alpha, iters)
    print("\n\ntheta_0=", theta[0, 0], "\ntheta_1=", theta[0, 1], "\ntheta_2=", theta[0, 2])

    print(sv.computeCost(X, y, theta))

    # 预测
    area = (2104 - mu[0, 0]) / std[0, 0]
    roomNum = (3 - mu[0, 1]) / std[0, 1]
    price = predictProfit(area, roomNum, theta)
    print("\n\n\npredict:2104, 3, 399900--->", price)
