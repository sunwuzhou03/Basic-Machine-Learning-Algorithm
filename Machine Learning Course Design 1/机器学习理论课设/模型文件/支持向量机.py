import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

# 设置 Matplotlib 的默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']


class SVM:
    def __init__(self, feature_size, C, alpha, N) -> None:
        '''
        w 超平面权重
        feature_size 特征大小, 即属性的个数
        b 偏置值
        alpha 学习率
        C 惩罚系数
        N 最大迭代次数
        '''
        self.feature_size = feature_size
        self.C = C
        self.alpha = alpha
        self.w = np.zeros(feature_size)
        self.b = 0
        self.N = N

    def forward(self, x):
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        y = self.forward(x)
        y_pred = y.copy()
        negative_index = np.where(y > 0)
        positive_index = np.where(y < 0)
        y_pred[negative_index] = 1
        y_pred[positive_index] = -1
        return np.array([y_pred]).reshape(-1, 1)

    #更新参数函数
    def train(self, data, label):
        # 随机梯度下降
        # r = random.randint(0, self.feature_size - 1)
        # predict = self.forward(data[r])
        # if label[r] * predict < 1:
        #     self.w = self.w - self.alpha * (self.w -
        #                                     self.C * label[r] * data[r])
        #     self.b = self.b - self.alpha * (-label[r] * self.C)
        # else:
        #     self.w = self.w - self.w * self.alpha

        #梯度下降
        for r in range(data.shape[0]):
            predict = self.forward(data[r])
            if label[r] * predict < 1:
                self.w = self.w - self.alpha * (self.w -
                                                self.C * label[r] * data[r])
                self.b = self.b - self.alpha * (-label[r] * self.C)
            else:
                self.w = self.w - self.w * self.alpha

    #训练函数，返回权重
    def fit(self, data, label):
        for _ in range(self.N):
            self.train(data, label)


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename).values.astype(float)
    return data[:, -1], data[:, :-1]


if __name__ == '__main__':
    #读取数据
    train_filename = '数据集//训练集//breast_cancer.csv'
    test_filename = '数据集//测试集//breast_cancer.csv'
    train_label, train_data = read_data(train_filename)
    test_label, test_data = read_data(test_filename)
    train_label = train_label.reshape((-1, 1))
    test_label = test_label.reshape((-1, 1))
    model = SVM(30, 100, 0.0001, 2000)
    model.fit(train_data, train_label)
    results = model.predict(test_data)
    print(np.sum(results == test_label) / test_label.shape[0])
