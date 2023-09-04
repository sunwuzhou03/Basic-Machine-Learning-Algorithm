import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Logistic:
    def __init__(self, feature_size, alpha, N) -> None:

        #初始化系数矩阵,系数权重为1
        self.theta = np.ones((feature_size, 1))

        #设置超参数
        self.alpha = alpha
        self.N = N

    #sigmoid函数
    def sigmoid(self, data):
        return (1 / (1 + np.exp(-data)))

    #模型预测函数
    def predict(self, data):
        #将大于0.5的变成1，小于0.5的变成0

        label_pred = np.where(self.forward(data) > 0.5, 1, 0)
        return label_pred

    #模型计算置信度
    def forward(self, data):
        label_pos = self.sigmoid(np.dot(data, self.theta))
        return label_pos

    #模型损失以及梯度计算函数
    def costFunction(self, data, label):
        m = len(label)
        #模型分类数据
        h = self.sigmoid(np.dot(data, self.theta))
        #将预测计算中不合格数据矫正
        one_index, zero_index = np.where(h >= 1), np.where(h <= 0)
        h[one_index] = 1 - 1e-10
        h[zero_index] = 1e-10
        #损失值
        loss = (-1 / m) * np.sum(label * np.log(h) +
                                 (1 - label) * np.log(1 - h))
        #梯度
        grad = (1 / m) * np.dot(data.T, (h - label))
        return loss, grad

    def fit(self, data, label):
        #开始使用梯度训练模型
        loss = []
        for i in range(self.N):
            per_loss, grad = self.costFunction(data, label)
            self.theta = self.theta - self.alpha * grad  #模型更新
            loss.append(per_loss)
        return loss


#数据标准化函数
def data_normlization(data):
    m, n = data.shape
    tempdata = data.copy()
    #对每一中属性的所有数据进行标准化
    for i in range(n):
        mu = np.mean(tempdata[:, i])
        sigma = np.std(tempdata[:, i])
        tempdata[:, i] = (tempdata[:, i] - mu) / sigma
    return tempdata


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename, header=None).values.astype(float)
    return data[:, -1], data[:, :-1]


if __name__ == '__main__':
    train_filename = '数据集//训练集//breast_cancer.csv'
    test_filename = '数据集//测试集//breast_cancer.csv'
    train_label, train_data = read_data(train_filename)
    test_label, test_data = read_data(test_filename)
    train_label = train_label.reshape((-1, 1))
    test_label = test_label.reshape((-1, 1))
    train_data = data_normlization(train_data)
    test_data = data_normlization(test_data)

    #改变训练集维度，为x加一维
    train_data = np.concatenate((np.ones(
        (train_data.shape[0], 1)), train_data),
                                axis=1)
    test_data = np.concatenate((np.ones((test_data.shape[0], 1)), test_data),
                               axis=1)

    data_num = train_data.shape[0]
    feature_size = train_data.shape[1]

    #设置超参数
    alpha = 0.1
    N = 500

    model = Logistic(feature_size, alpha, N)
    loss = model.fit(train_data, train_label)
    results = model.predict(test_data)
    print(np.sum(results == test_label) / test_label.shape[0])
    plt.plot(loss)
    plt.show()
