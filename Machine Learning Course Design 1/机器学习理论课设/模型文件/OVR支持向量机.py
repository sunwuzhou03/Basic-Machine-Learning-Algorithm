from 支持向量机 import SVM
import pandas as pd
import numpy as np
import os
import pickle


#OVR多分类
class OVRSVM:
    def __init__(self, input_dim, output_dim, C, alpha, N) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.C = C
        self.N = N
        self.models = [SVM(input_dim, C, alpha, N) for _ in range(output_dim)]

    def reset(self):
        self.models = [
            SVM(self.input_dim, self.C, self.alpha, self.N)
            for _ in range(self.output_dim)
        ]

    def fit(self, data, label):
        for i, model in enumerate(self.models):
            label_tran = label.copy()
            #print(label_tran)
            negative_index = np.where(label == i)
            positive_index = np.where(label != i)
            label_tran[negative_index] = 1
            label_tran[positive_index] = -1
            model.fit(data, label_tran)

    def predict(self, data):
        label_pred = []
        for single_data in data:
            label = None
            label_pos = 0
            for i, model in enumerate(self.models):
                pos = model.forward(single_data.reshape(1, -1)).item()
                if pos >= label_pos:
                    label = i
                    label_pos = pos
            label_pred.append(label)
        return np.array(label_pred).reshape(-1, 1)

    def save_params(self, filename):
        '''
        模型参数保存函数
        filename   模型参数保存路径
        '''
        path = os.makedirs(f"{filename}", exist_ok=True)
        path = f"{filename}"
        with open(f"{path}\params.pkl", 'wb') as f:
            pickle.dump(self.models, f)

    def load_params(self, filename):
        '''
        模型参数加载函数
        filename   模型参数加载路径
        '''
        path = os.makedirs(f"{filename}", exist_ok=True)
        path = f"{filename}"
        with open(f"{path}\params.pkl", 'rb') as f:
            self.models = pickle.load(f)


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


if __name__ == '__main__':
    # train_filename = '数据集\data_banknote_authentication.csv'
    # test_filename = '数据集\data_banknote_authentication.csv'
    train_filename = '数据集//训练集//dermatology.csv'
    test_filename = '数据集//测试集//dermatology.csv'

    model = OVRSVM(34, 6, 100, 0.0001, 3000)
    # train_filename = 'experiment_05_training_set.csv'
    # test_filename = 'experiment_05_testing_set.csv'
    train_label, train_data = read_data(train_filename)
    test_label, test_data = read_data(test_filename)
    train_label = train_label.reshape((-1, 1))
    test_label = test_label.reshape((-1, 1))
    train_data = data_normalization(train_data)
    test_data = data_normalization(test_data)
    model.fit(train_data, train_label)
    model.save_params('OVR支持向量机//dermatology')
    model.load_params('OVR支持向量机//dermatology')
    results = model.predict(test_data)
    print(np.sum(results == test_label) / test_label.shape[0])
