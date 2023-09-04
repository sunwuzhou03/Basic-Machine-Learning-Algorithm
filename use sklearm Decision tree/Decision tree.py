import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn import tree

#数据标准化函数
def data_score_stdlize(data):
    m, n = data.shape
    tempdata = data.copy()
    #对每一中属性的所有数据进行标准化
    for i in range(n):
        mu = np.mean(tempdata[:, i])
        sigma = np.std(tempdata[:, i])
        tempdata[:, i] = (tempdata[:, i] - mu) / sigma
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
    return data[:, :-1], data[:, -1]


#读取数据以及数据标准化
train_filename = 'experiment_04_training_set.csv'
test_filename = 'experiment_04_testing_set.csv'
x_train, y_train = read_data(train_filename)
x_test, y_test = read_data(train_filename)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
x_train_std = data_score_stdlize(x_train)
x_test_std = data_score_stdlize(x_test)

#使用sklearn对数据进行决策树划分
for item in ["gini","entropy"]:#两种划分标准
    print(f"{item}:")
    for i in range(1,4):
        clf=tree.DecisionTreeClassifier(criterion=item,random_state=1,max_depth=i)
        clf=clf.fit(x_train_std,y_train)#训练
        score=clf.score(x_test_std,y_test)#计算准确率
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        tree.plot_tree(clf,filled=True)#使用库函数生成决策树图片
        fig.savefig(f'criterion_{item}_max_depth_{i}.png')#保存决策树图片
        print(score,end='    ')#输出结果
    print()
