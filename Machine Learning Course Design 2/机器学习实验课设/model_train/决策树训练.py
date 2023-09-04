from sklearn import tree
import numpy as np
import csv
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import random


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename, header=None).values.astype(float)
    return data


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


#读取数据以及数据标准化
data = read_data('data//MNIST//raw//mnist_train.csv')
data_copy = data.copy()
data, label = data_copy[:, 1:] / 255, data[:, 0]

print("训练开始！")
clf = tree.DecisionTreeClassifier()
clf.fit(data, label)
joblib.dump(clf, 'save_model\\tree.pkl')
#load model
# classifier = joblib.load('save_model\\classifier.pkl')

# 以文字形式输出树
# text_representation = tree.export_text(classifier)
# print(text_representation)

# # 用图片画出
# plt.figure(figsize=(30, 10), facecolor='g')  #
# a = tree.plot_tree(classifier, rounded=True, filled=True, fontsize=14)
# plt.show()

print("训练结束！")