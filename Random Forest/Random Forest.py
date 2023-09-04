import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename).values.astype(float)
    return data


#自助采样
def bootstrap_sample(data, num_samples):
    samples = []
    n = len(data)
    for i in range(num_samples):
        sample = [data[random.randint(0, n - 1)] for j in range(n)]
        samples.append(sample)
    return samples


#Bagging方法
def Bagging(data, n_estimators):
    num_samples = n_estimators
    samples = bootstrap_sample(data, num_samples)
    hlist = []
    for t in range(n_estimators):  #训练基模型
        train_data = np.array(samples[t])[:, 1:] / 255.0  #将数据变成[0,1]的范围
        train_label = np.array(samples[t])[:, 0].reshape(-1, 1)
        ht = DecisionTreeClassifier(criterion='entropy', max_features=50)
        ht.fit(train_data, train_label)
        hlist.append(ht)
    return hlist


def predict(hlist, data):
    label_pred = np.zeros(10)
    for ht in hlist:
        #print(ht.predict(data.reshape(1, -1)))
        label_pred[int(ht.predict(data.reshape(1, -1))[0])] += 1
    return np.argmax(label_pred)


train = True  #是否训练
#读取数据以及数据标准化
train_filename = 'experiment_09_training_set.csv'
test_filename = 'experiment_09_testing_set.csv'
train_data = read_data(train_filename)
test_data = read_data(test_filename)
test_label = np.array(test_data)[:, 0].reshape(-1, 1)
test_data = np.array(test_data)[:, 1:] / 255.0

#开始训练
T_list = list(range(1, 21))
acc_list = []
h_list = []
for T in T_list:
    h_list = hlist = Bagging(train_data, T)
    correct_num = 0
    wrong_num = 0
    for x, y in zip(test_data, test_label):  #测试
        y_pred = predict(hlist, x)
        if y_pred == y:
            correct_num += 1
        else:
            wrong_num += 1
    acc = correct_num / (correct_num + wrong_num)
    acc_list.append(acc)
    print('T =', T, 'Accuracy =%.2f ' % acc)

#绘图
plt.plot(T_list, acc_list, marker='o')
plt.xlabel('Number of Base Models T')
plt.ylabel('Accuracy on Test Set')
plt.xticks(T_list)
plt.show()