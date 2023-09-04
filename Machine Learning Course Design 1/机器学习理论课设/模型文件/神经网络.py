import numpy as np
import pandas as pd
import pickle
import os


#神经网络网络类
class Network:
    def __init__(self, input_dim, hidden_dim, output_dim, lr) -> None:
        '''
        类初始化函数
        batch_size  每次更新的数据量
        input_dim   输入层大小
        hidden_dim  隐藏层大小
        output_dim  输出层大小
        activation  激活函数
        采用标准正态分布初始化模型参数
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = 0
        self.weight2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = 0
        self.output1 = 0
        self.output2 = 0
        #print(f"weight1shape={self.weight1.shape}")

    def reset(self):
        self.weight1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.bias1 = 0
        self.weight2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.bias2 = 0
        self.output1 = 0
        self.output2 = 0

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

        #权重更新函数
    def upadate(self, data, label):
        _, _ = self.classification(data)
        J, delta_v, delta_gamma, delta_w, delta_theta = self.loss_function(
            data, label)
        self.weight1 = np.add(self.lr * delta_v, self.weight1)
        self.weight2 = np.add(self.lr * delta_w, self.weight2)
        return J

    #sigmoid激活函数
    def activation(self, x):
        return 1 / (np.exp(-x) + 1)

    def loss_function(self, data, label):
        '''
        data   待分类数据
        label  数据标签，即该数据的种类
        返回损失函数以及各参数的梯度
        '''
        label_array = np.zeros((data.shape[0], self.output_dim))
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


class Neural_Network:
    '''
    input_dim: 输入层大小
    hidden_dim: 隐藏层大小
    output_dim: 输出层大小
    lr: 学习率
    batch_size: 批大小
    N: 训练轮数  
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, lr, batch_size,
                 N) -> None:
        self.model = Network(input_dim, hidden_dim, output_dim, lr)
        self.batch_size = batch_size
        self.N = N

    #模型训练（测试）函数
    def train_model(self, data, label):
        '''
        model    模型
        N        最大迭代次数
        lr       模型学习率
        data     训练数据
        label    训练数据对应的标签
        filename 保存（导入）模型参数的文件名
        train    True为训练模式，False为测试模式
        '''
        per_loss = 0
        for single_data, single_label in zip(data, label):
            # print(label.shape)
            J = self.model.upadate(single_data, single_label)
            per_loss += np.sum(J)
        return per_loss

    def reset(self):
        self.model.reset()

    def fit(self, data, label):
        data = dataloader(data, self.batch_size)
        label = dataloader(label, self.batch_size)
        for _ in range(self.N):
            self.train_model(data, label)

    def predict(self, data):
        label_pred, _ = self.model.classification(data)
        return label_pred

    def save_params(self, filename):
        '''
        模型参数保存函数
        filename   模型参数保存路径
        '''
        path = os.makedirs(f"{filename}", exist_ok=True)
        path = f"{filename}"
        with open(f"{path}\params.pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def load_params(self, filename):
        '''
        模型参数加载函数
        filename   模型参数加载路径
        '''
        path = os.makedirs(f"{filename}", exist_ok=True)
        path = f"{filename}"
        with open(f"{path}\params.pkl", 'rb') as f:
            self.model = pickle.load(f)


#数据标准化函数
def data_normalization(data):
    data_copy = data.copy()
    m, n = data.shape
    for col in range(n):
        min_value = np.min(data_copy[:, col])
        max_value = np.max(data_copy[:, col])
        value = (max_value - min_value) if (max_value - min_value) != 0 else 1
        data_copy[:, col] = (data_copy[:, col] - min_value) / value
    return data_copy


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename).values.astype(float)
    return data[:, -1], data[:, :-1]


def dataloader(data, batch_size):
    dataset = []
    for i in range(int(data.shape[0] / batch_size) + 1):
        if (i + 1) * batch_size:
            dataset.append(np.array(data[i * batch_size:(i + 1) * batch_size]))
        else:
            dataset.append(np.array(data[i * batch_size:]))

    return dataset


if __name__ == '__main__':
    filename = "神经网络/dermatology"
    #训练
    loss = []
    train = True  #是否训练
    #读取数据以及数据标准化
    train_filename = '数据集//数据集//训练集//dermatology.csv'
    test_filename = '数据集//数据集//测试集//dermatology.csv'
    y_train, x_train = read_data(train_filename)
    y_test, x_test = read_data(test_filename)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    x_train = data_normalization(x_train)
    x_test = data_normalization(x_test)
    model = Neural_Network(34, 22, 6, 0.3, 4, 500)
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    print(np.sum(results == y_test) / y_test.shape[0])
