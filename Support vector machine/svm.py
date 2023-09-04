import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.font_manager import FontProperties

# 设置 Matplotlib 的默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']


#读取数据函数，返回属性和标签
def read_data(filename):
    data = []
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        #跳过第一行
        header = next(csvreader)
        data = [row for row in csvreader]
    data = np.array(data).astype(float)
    return data[:, -1], data[:, :-1]


def forward(w, b, x):
    return np.dot(x, w) + b


#更新参数svm函数
def svm(C, alpha, w, b, data, label):
    r = random.randint(0, row - 1)
    predict = forward(w, b, data[r])
    if label[r] * predict < 1:
        w = w - alpha * (w - C * label[r] * data[r])
        b = b - alpha * (-label[r] * C)
    else:
        w = w - w * alpha
    return w, b


#测试函数，返回精度
def test(w, b, x, y):
    test_data, test_label = x, y
    pre_list = []
    for item in test_data:
        y_pre = forward(w, b, item)
        if y_pre[0] > 0:
            pre_list.append(1)
        else:
            pre_list.append(-1)
    count = 0
    for i in range(len(pre_list)):
        if (pre_list[i] == test_label[i][0]):
            count += 1
    return count / len(pre_list)


#训练函数，返回权重
def train(w, b, alpha, C):
    global x_train_std, x_test_std, y_train, y_test
    w, b = svm(C, alpha, w, b, x_train, y_train)
    return w, b


#读取数据
train_filename = 'experiment_06_training_set.csv'
test_filename = 'experiment_06_testing_set.csv'
y_train, x_train = read_data(train_filename)
y_test, x_test = read_data(test_filename)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
train_acc_vec = []
test_acc_vec = []
row, col = np.shape(x_train)
'''
w 超平面权重
b 偏置值
alpha 学习率
C 惩罚系数
N 最大迭代次数
'''

w = np.zeros(col)
b = 0.0
alpha = 0.001
C = 100
N = 2000
bestw = w
bestb = b
bestacc = 0

#开始训练
for i in range(N):
    w, b = train(w, b, alpha, C)
    test_acc = test(w, b, x_test, y_test)
    test_acc_vec.append(test_acc)
    train_acc = test(w, b, x_train, y_train)
    train_acc_vec.append(train_acc)

#超平面方程
x_ = np.arange(-2, 6.5, 1)
y_ = -w[0] / w[1] * x_ - b / w[1]

#在二维坐标系中表示测试集各个点以及超平面图
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm')  #画出测试集样例点
plt.plot(x_, y_, c='red')  #画出超平面
plt.title('分类超平面图')
plt.xlim([-2, 6.5])
plt.ylim([-2, 6.5])
plt.xlabel(u'x')
plt.ylabel(u'y')
plt.show()

# 生成两组 x 和 y 值
x = np.arange(0, N)
y1 = train_acc_vec  #训练集精度迭代向量
y2 = test_acc_vec  #测试集精度迭代向量

#绘制训练集精度迭代曲线
plt.plot(x, y1, label='train_acc')
plt.title('训练集精度迭代曲线')
plt.xlabel(u'epoch')
plt.ylabel(u'accuracy')
plt.show()
plt.plot(x, y2, label='test_acc', c='green')

#绘制测试集精度迭代曲线
plt.title('测试集集精度迭代曲线')
plt.xlabel(u'epoch')
plt.ylabel(u'accuracy')
plt.show()

test_acc = test(w, b, x_test, y_test)
#输出测试集上的精度，以及支持向量机表达式
print(f"测试集上精度{test_acc}，该分类超平面为 y = ({-w[0] / w[1]}) * x + ({-b[0] / w[1]})")
