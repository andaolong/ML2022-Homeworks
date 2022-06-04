# 用python实现，吴恩达机器学习慕课-实验一-中的单变量回归任务
# 将之前用Octave写的行数转化为python实现一下就好了，思想是一样的
# 2022年6月04日8:20:58：

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1.读取数据
def importDataFromFile(path):
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    # print(data.head())      # 打印观察一下前几行,默认输出前5行
    # print(data.describe())  # 打印观察一下该表格的信息值，如数目平均值标准差等

    # 绘制一下图像观察一下
    # data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    # plt.show()

    data.insert(0, 'Ones', 1)  # 训练集前面插入一列作为截距项x_0
    cols = data.shape[1]  # 列数
    X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
    y = data.iloc[:, cols - 1:cols]  # X是所有行，最后一列

    # 将pandas读取进来的数据转换成numpy的矩阵形式, 下面的X，y，theta用来返回
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # theta = np.matrix(np.array([0, 0]))
    theta = np.zeros((1, cols - 1))

    return X, y, theta, data


# 2.定义-计算损失函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 3.定义-梯度下降函数
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    # parameters = int(theta.ravel().shape[1])
    parameters = theta.size

    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# 4.绘制图像
def plotImage(data, theta):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0, 0] + (theta[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def plotCost(cost, iters):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


# 5.预测
def predictProfit(population, theta):
    profit = theta[0, 0] + theta[0, 1] * population;
    return profit


# 主函数入口
if __name__ == '__main__':
    # 指定数据路径
    path = 'ex1data1.txt'
    # 定义学习率和迭代次数-超参数
    alpha = 0.01
    iters = 1000

    # 读入数据, 在此X为m行n+1列，也就是m行2列，y是m行1列，theta是1行2列
    X, y, theta, data = importDataFromFile(path)

    # 进行梯度下降，训练求解
    theta, cost = gradientDescent(X, y, theta, alpha, iters)
    print("theta_0=", theta[0, 0], "\n", "theta_1=", theta[0, 1])

    # 绘制结果图像
    plotImage(data, theta)

    # 绘制损失随迭代次数下降的曲线
    plotCost(cost, iters)

    # 给定一个population值预测利润值, 人口单位为/万人，利润单位也是/万美元
    population = 5.5277
    profit = predictProfit(population, theta)
    print("人口为", population, "万人的城市，新增一辆小吃车的利润为", profit, "万美元")
