from sklearn.svm import LinearSVC
import csv
import numpy as np
import joblib
import pandas as pd
import numpy as np
import random


def split_dataset(dataset, ratio, random_state=42):
    dataset_copy = np.array(dataset).copy()
    train_size = int(len(dataset_copy) * ratio)
    train_set = []
    valid_set = list(dataset_copy)
    np.random.seed(random_state)
    while len(train_set) < train_size:
        index = random.randrange(len(valid_set))
        train_set.append(valid_set.pop(index))
    return np.array(train_set), np.array(valid_set)


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename, header=None).values.astype(float)
    return data


#读取数据以及数据标准化
data = read_data('data//MNIST//raw//mnist_train.csv')
trainData, validData = split_dataset(data, 0.8)
data, label = data.copy()[:, 1:] / 255, data.copy()[:, 0].reshape((-1, 1))
train_data, train_label = trainData[:, 1:] / 255, trainData[:, 0].reshape(
    (-1, 1))
valid_data, valid_label = validData[:, 1:] / 255, validData[:, 0].reshape(
    (-1, 1))
## 模型训练和预测
#加上dual=False,不然会报错 ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.

svc = LinearSVC(dual=False)
svc.fit(train_data, train_label.T.ravel())

print('Accuracy of Linear SVC:', svc.score(valid_data, valid_label.T.ravel()))
