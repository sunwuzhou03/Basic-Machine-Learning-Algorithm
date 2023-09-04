import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


#数据标准化函数
def data_score_stdlize(data):
    tempdata = data.copy() / 255.0

    return tempdata


#读取数据函数，返回属性和标签
def read_data(filename):
    data = []
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        #跳过第一行
        header = next(csvreader)
        data = [row for row in csvreader]
    data = np.array(data).astype(float)
    return data[:, 0], data[:, 1:]


#sigmoid激活函数
def activation(x):
    return 1 / (np.exp(-x) + 1)


#权重更新函数
def upadate(model, data, label, lr):
    _, _ = model.classification(data)
    J, delta_v, delta_gamma, delta_w, delta_theta = model.loss_function(
        data, label)
    model.weight1 = np.add(lr * delta_v, model.weight1)
    model.bias1 = np.add(lr * delta_gamma, model.bias1)
    model.weight2 = np.add(lr * delta_w, model.weight2)
    model.bias2 = np.add(lr * delta_theta, model.bias2)
    return J


#模型训练（测试）函数
def train_model(model, lr, data, label, filename, train=True):
    '''
    model    模型
    epoch    最大迭代次数
    lr       模型学习率
    data     训练数据
    label    训练数据对应的标签
    filename 保存（导入）模型参数的文件名
    train    True为训练模式，False为测试模式
    '''
    epoch_loss = 0
    for x, y in zip(data, label):
        J = upadate(model, x, y, lr)
        epoch_loss += np.sum(J)
    return epoch_loss


def test_model(model, lr, data, label, filename):
    #导入模型参数
    model.load_params(f"{filename}\lr_{lr}")
    correct_num = 0
    total_num = x_test.shape[0]
    for x, y in zip(data, label):
        y_pred, output = model.classification(x)
        if y_pred == y:
            correct_num += 1
    return correct_num / total_num  #返回学习率


#神经网络网络类
class neural_networks:
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim,
                 activation) -> None:
        '''
        类初始化函数
        batch_size  每次更新的数据量
        input_dim   输入层大小
        hidden_dim  隐藏层大小
        output_dim  输出层大小
        activation  激活函数
        采用标准正态分布初始化模型参数
        '''
        self.weight1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.random.randn(1, 12)
        self.weight2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.random.randn(1, 10)
        self.output1 = 0
        self.output2 = 0
        self.activation = activation

    def save_params(self, filename):
        '''
        模型参数保存函数
        filename   模型参数保存路径
        '''
        np.savetxt(f"{filename}\\weight1.csv", self.weight1, delimiter=',')
        np.savetxt(f"{filename}\\weight2.csv", self.weight2, delimiter=',')
        np.savetxt(f"{filename}\\bias1.csv", self.bias1, delimiter=',')
        np.savetxt(f"{filename}\\bias2.csv", self.bias2, delimiter=',')

    def load_params(self, filename):
        '''
        模型参数加载函数
        filename   模型参数加载路径
        '''
        self.bias1 = np.array([
            np.loadtxt(open(f"{filename}\\bias1.csv", "rb"),
                       delimiter=",",
                       skiprows=0)
        ])
        self.bias2 = np.array([
            np.loadtxt(open(f"{filename}\\bias2.csv", "rb"),
                       delimiter=",",
                       skiprows=0)
        ])
        self.weight1 = np.loadtxt(open(f"{filename}\\weight1.csv", "rb"),
                                  delimiter=",",
                                  skiprows=0)
        self.weight2 = np.loadtxt(open(f"{filename}\\weight2.csv", "rb"),
                                  delimiter=",",
                                  skiprows=0)

    def classification(self, data):
        '''
        分类函数
        data 待分类数据
        返回该数据预测种类以及所有种类的分布
        '''
        self.output1 = self.activation(np.dot(data, self.weight1) + self.bias1)
        self.output2 = self.activation(
            np.dot(self.output1, self.weight2) + self.bias2)
        return np.argmax(self.output2), self.output2

    def loss_function(self, data, label):
        '''
        data   待分类数据
        label  数据标签，即该数据的种类
        返回损失函数以及各参数的梯度
        '''
        label_array = np.zeros((1, 10))
        label_array[0, int(label)] = 1
        g = np.multiply(self.output2, (1 - self.output2))
        g = np.multiply(g, label_array - self.output2)
        e = np.multiply(self.output1, (1 - self.output1))
        e = np.multiply(e, np.dot(self.weight2, g.T).reshape(-1))
        e = np.array(e)
        g = np.array(g)
        delta_w = np.dot(self.output1.T, g)
        delta_theta = -g  #隐藏层偏置值
        delta_v = np.dot(np.array([data]).T, e)
        delta_gamma = -e  #输入层偏置值
        pred = self.output2
        J = 0.5 * (label_array - pred)**2
        return J, delta_v, delta_gamma, delta_w, delta_theta


#读取数据以及数据标准化
train_filename = 'experiment_05_training_set.csv'
test_filename = 'experiment_05_testing_set.csv'
y_train, x_train = read_data(train_filename)
y_test, x_test = read_data(test_filename)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
x_train_std = data_score_stdlize(x_train)
x_test_std = data_score_stdlize(x_test)
batch_size = 1
input_dim = x_train_std.shape[1]
hidden_dim = 12
output_dim = 10
lr = [0.001, 0.005, 0.01]
epoch = 300
filename = "params"
#训练
loss = []

train = True  #是否训练

if train:
    for lri in lr:
        model = neural_networks(batch_size, input_dim, hidden_dim, output_dim,
                                activation)
        p_loss = []
        for i in range(epoch):
            per_loss = train_model(model=model,
                                   lr=lri,
                                   data=x_train_std,
                                   label=y_train,
                                   filename=filename,
                                   train=True)
            if i % 10 == 0:  #每十轮保存一次参数，将参数保存为csv文件
                model.save_params(f"{filename}\lr_{lri}")
                print(f"学习率为{lri}的模型训练第{i}轮")
                accuracy = test_model(
                    model=model,
                    lr=lri,
                    data=x_test_std,
                    label=y_test,
                    filename=filename,
                )
                print(f"学习率为{lri}的模型在测试集上准确率为{accuracy}")
            print(per_loss)
            p_loss.append(per_loss)
        loss.append(p_loss)

        fig, ax = plt.subplots()
        ax.set_title(f"lr_{lri}_model_loss_curve")
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss per epoch")
        ax.plot(np.arange(0, epoch),
                p_loss,
                label=f"lr_{lri}_model_loss_curve")
        plt.savefig(f"lr_{lri}_model_loss.png")
        plt.show(block=False)

#测试
for lri in lr:
    model = neural_networks(batch_size, input_dim, hidden_dim, output_dim,
                            activation)
    accuracy = test_model(
        model=model,
        lr=lri,
        data=x_test_std,
        label=y_test,
        filename=filename,
    )
    print(f"学习率为{lri}的模型在测试集上准确率为{accuracy}")

if train:
    index = [np.arange(0, epoch)] * 3
    for i in range(3):
        plt.plot(loss[i], label=f"lr_{lr[i]}_model_loss_curve")
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        plt.xticks(fontproperties='Times New Roman', fontsize=10)
        plt.xticks(fontproperties='Times New Roman', fontsize=10)
        plt.xlabel(u'epochs')
        plt.ylabel(u'loss per epoch')
        plt.legend(loc=1, prop=font)
    plt.title("3_model_loss_curve")
    plt.savefig("3_model_loss_curve")
    plt.show(block=False)
