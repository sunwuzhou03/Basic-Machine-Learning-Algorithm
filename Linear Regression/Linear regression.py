import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


#使用面向对象方法定义线性回归模型
class LineRegression:
    def __init__(self, x, y):
        self.train_num = x.shape[0]  #训练数据量
        self.train_feature = x.shape[1]  #特征量
        self.x = x
        self.y = y
        #随机生成系数矩阵
        self.w = np.random.randn(self.train_feature, 1)
        #偏移量初始化为0
        self.b = 0

    def update(self, learing_rate, dw, db):
        #梯度下降更新
        self.w = self.w - learing_rate * dw
        self.b = self.b - learing_rate * db

    def train(self, learning_rate, epochs):
        loss = []
        for i in range(epochs):
            #计算预测值
            y_prediction = self.prediction(self.x)
            #计算梯度
            dw = np.dot(self.x.transpose(),
                        (y_prediction - self.y)) / self.train_num
            db = np.sum((y_prediction - self.y)) / self.train_num
            #计算每一轮的损失值
            per_loss = np.sum((y_prediction - self.y)**2) / self.train_num
            self.update(learning_rate, dw, db)
            loss.append(per_loss)
        return loss

    #预测函数
    def prediction(self, x):
        return np.dot(x, self.w) + self.b


#计算性能衡量指标
def CalculateError(y_true, y_predicion):
    MSE = np.sum((y_predicion - y_true)**2, axis=0) / y_predicion.shape[0]
    RMSE = (np.sum(
        (y_predicion - y_true)**2, axis=0) / y_predicion.shape[0])**(0.5)
    MAE = np.sum(abs(y_predicion - y_true), axis=0) / y_predicion.shape[0]
    return MSE.item(), RMSE.item(), MAE.item()


#load the dataser of the sklearn
diabetes = load_diabetes()
data, target = diabetes.data, diabetes.target
x, y = shuffle(data, target, random_state=13)
offset = int(x.shape[0] * 0.8)
x_train, y_train = x[:offset], y[:offset]
x_test, y_test = x[offset:], y[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

L1 = LineRegression(x_train, y_train)
L2 = LineRegression(x_train, y_train)
L3 = LineRegression(x_train, y_train)
#模型1，2，3学习率分别为0.1，0.01，0.001
loss1 = L1.train(0.1, 200000)
loss2 = L2.train(0.01, 200000)
loss3 = L3.train(0.001, 200000)

y_true = y_test  #真实值
#计算不同学习率下的预测值
prediction_1 = L1.prediction(x_test)
prediction_2 = L2.prediction(x_test)
prediction_3 = L3.prediction(x_test)

#计算MSE，RMSE,MAE
MSE1, RMSE1, MAE1 = CalculateError(y_true, prediction_1)
MSE2, RMSE2, MAE2 = CalculateError(y_true, prediction_2)
MSE3, RMSE3, MAE3 = CalculateError(y_true, prediction_3)

#输出结果
print("MSE of prediction 1:", MSE1)
print("MSE of prediction 2:", MSE2)
print("MSE of prediction 3:", MSE3)

print("RMSE of prediction 1:", RMSE1)
print("RMSE of prediction 2:", RMSE2)
print("RMSE of prediction 3:", RMSE3)

print("MAE of prediction 1:", MAE1)
print("MAE of prediction 2:", MAE2)
print("MAE of prediction 3:", MAE3)

#0到200000步
index = np.arange(0, 200000, 1)

#绘制损失曲线图
plt.plot(index,
         loss1,
         c='blue',
         marker='o',
         linestyle='-',
         label='lr=0.1_loss1')
plt.plot(index,
         loss2,
         c='green',
         marker='*',
         linestyle='--',
         label='lr=0.01_loss2')
plt.plot(index,
         loss3,
         c='red',
         marker='+',
         linestyle=':',
         label='lr=0.001_loss3')

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.xticks(fontproperties='Times New Roman', fontsize=10)
plt.xticks(fontproperties='Times New Roman', fontsize=10)

#写出横纵坐标标签
plt.xlabel(u'epoch')
plt.ylabel(u'loss each epoch')

plt.legend(loc=1, prop=font)
plt.show()
