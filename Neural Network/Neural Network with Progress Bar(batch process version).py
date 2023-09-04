import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import os
from tqdm import trange


#数据标准化函数
def data_normlization(data):
    data_copy = data.copy()
    m, n = data.shape
    for col in range(n):
        min_value = np.min(data_copy[:, col])
        max_value = np.max(data_copy[:, col])
        value = (max_value - min_value) if (max_value - min_value) != 0 else 1
        data_copy[:, col] = (max_value - data_copy[:, col]) / value
    return data_copy


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename).values.astype(float)
    return data[:, -1], data[:, :-1]


def dataloader(data, bacth_size):
    dataset = []
    for i in range(int(data.shape[0] / batch_size) + 1):
        if (i + 1) * bacth_size:
            dataset.append(np.array(data[i * batch_size:(i + 1) * bacth_size]))
        else:
            dataset.append(np.array(data[i * batch_size:]))

    return dataset


#sigmoid激活函数
def activation(x):
    return 1 / (np.exp(-x) + 1)


#权重更新函数
def upadate(model, data, label, lr):
    _, _ = model.classification(data)
    J, delta_v, delta_gamma, delta_w, delta_theta = model.loss_function(
        data, label)
    model.weight1 = np.add(lr * delta_v, model.weight1)
    model.weight2 = np.add(lr * delta_w, model.weight2)
    return J


#模型训练（测试）函数
def train_model(model, lr, datas, labels, filename):
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
    for data, label in zip(datas, labels):
        # print(label.shape)
        J = upadate(model, data, label, lr)
        epoch_loss += np.sum(J)
    return epoch_loss


def test_model(model, lr, datas, labels, filename):
    #导入模型参数
    model.load_params(f"{filename}")
    correct_num = 0
    wrong_num = 0
    for data, label in zip(datas, labels):
        label_pred, _ = model.classification(data)
        correct_num += np.sum(label_pred == label)
        wrong_num += np.sum(label_pred != label)
    return correct_num / (correct_num + wrong_num)  #返回正确率


#神经网络网络类
class neural_networks:
    def __init__(self, input_dim, hidden_dim, output_dim, activation) -> None:
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
        self.bias1 = 0
        self.weight2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = 0
        self.output1 = 0
        self.output2 = 0
        self.activation = activation
        #print(f"weight1shape={self.weight1.shape}")

    def save_params(self, filename):
        '''
        模型参数保存函数
        filename   模型参数保存路径
        '''
        path = os.makedirs(f"{filename}", exist_ok=True)
        path = f"{filename}"
        with open(f"{path}\weight1.pkl", 'wb') as f:
            pickle.dump(self.weight1, f)
        with open(f"{path}\weight2.pkl", 'wb') as f:
            pickle.dump(self.weight2, f)

    def load_params(self, filename):
        '''
        模型参数加载函数
        filename   模型参数加载路径
        '''
        path = os.makedirs(f"{filename}", exist_ok=True)
        path = f"{filename}"
        with open(f"{path}\weight1.pkl", 'rb') as f:
            self.weight1 = pickle.load(f)
        with open(f"{path}\weight2.pkl", 'rb') as f:
            self.weight2 = pickle.load(f)

    def classification(self, data):
        '''
        分类函数
        data 待分类数据
        返回该数据预测种类以及所有种类的分布
        '''
        self.output1 = self.activation(np.dot(data, self.weight1))
        self.output2 = self.activation(np.dot(self.output1, self.weight2))
        np.expand_dims(np.argmax(self.output2, axis=1), axis=0)
        return np.expand_dims(np.argmax(self.output2, axis=1),
                              axis=0).reshape(-1, 1), self.output2

    def loss_function(self, data, label):
        '''
        data   待分类数据
        label  数据标签，即该数据的种类
        返回损失函数以及各参数的梯度
        '''
        label_array = np.zeros((data.shape[0], output_dim))
        for (x, y) in [(i, int(label[i])) for i in range(data.shape[0])]:
            label_array[x, y] = 1

        if label_array.shape != self.output2.shape:
            print(
                f"label_array.shape,self.output2.shape={label_array.shape,self.output2.shape}"
            )
        g = self.output2 * (1 - self.output2) * (label_array - self.output2)
        e = self.output1 * (1 - self.output1) * (np.dot(self.weight2, g.T)).T
        delta_w = np.dot(self.output1.T, g)
        delta_theta = -g  #隐藏层偏置值
        delta_v = np.dot(np.array(data).T, e)
        delta_gamma = -e  #输入层偏置值
        pred = self.output2
        J = 0.5 * (label_array - pred)**2  #均方损失函数
        return J, delta_v, delta_gamma, delta_w, delta_theta


batch_size = 64
input_dim = 784
hidden_dim = 12
output_dim = 6
lr = 0.3
epoch = 300
filename = "神经网络/dermatology"
#训练
loss = []

train = True  #是否训练
#读取数据以及数据标准化
train_filename = 'experiment_05_training_set.csv'
test_filename = 'experiment_05_testing_set.csv'
y_train, x_train = read_data(train_filename)
y_test, x_test = read_data(test_filename)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
x_train = data_normlization(x_train)
x_test = data_normlization(x_test)
y_train = dataloader(y_train, batch_size)
y_test = dataloader(y_test, batch_size)
x_train = dataloader(x_train, batch_size)
x_test = dataloader(x_test, batch_size)

if train:

    model = neural_networks(input_dim, hidden_dim, output_dim, activation)
    p_loss = []

    processBar = trange(epoch, unit='step')

    for i in range(epoch):
        per_loss = train_model(
            model=model,
            lr=lr,
            datas=x_train,
            labels=y_train,
            filename=filename,
        )
        if i % 10 == 0:  #每十轮保存一次参数，将参数保存为pkl文件
            model.save_params(f"{filename}")
            accuracy = test_model(
                model=model,
                lr=lr,
                datas=x_test,
                labels=y_test,
                filename=filename,
            )
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (i, epoch, per_loss, accuracy))
        processBar.update(1)
        p_loss.append(per_loss)
    loss.append(p_loss)

    # fig, ax = plt.subplots()
    # ax.set_title(f"lr_{lr}_model_loss_curve")
    # ax.set_xlabel("epochs")
    # ax.set_ylabel("loss per epoch")
    # ax.plot(np.arange(0, epoch), p_loss, label=f"lr_{lr}_model_loss_curve")
    # plt.savefig(f"lr_{lr}_model_loss.png")
    # plt.show(block=False)
processBar.close()
