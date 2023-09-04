import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


#数据读取函数
def read_data(filename):
    data = pd.read_csv(filename).values.astype(float)
    return data[:, :-1], data[:, -1]


#预测函数
def prediction(h_list, alpha_list, data):
    m, n = data.shape
    y_pred = np.zeros(m)
    for i, h in enumerate(h_list):
        y_pred += alpha_list[i] * h.predict(data)
    y_pred = np.sign(y_pred)
    return y_pred


#AdaBoost算法
def AdaBoost(x_train, y_train, T):
    N = len(y_train)
    w = np.ones(N) / N
    h_list = []
    alpha_list = []
    for t in range(T):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        tree.fit(x_train, y_train, sample_weight=w)
        y_pred = tree.predict(x_train)
        err = np.sum(w * (y_pred != y_train))
        if err>0.5:
            break
        alpha = 0.5 * np.log((1 - err) / err)
        w = w * np.exp(-alpha * y_train * y_pred)
        w = w / np.sum(w)
        h_list.append(tree)
        alpha_list.append(alpha)

    return h_list, alpha_list


#分类边界绘制函数
def plot_boundary(x, y, models, alpha):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, h),
                         np.arange(x[:, 1].min() - 1, x[:, 1].max() + 1, h))
    Z = (prediction(models, alpha, np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('AdaBoost Classification Boundary')
    plt.show()


train_filename = 'experiment_08_training_set.csv'
test_filename = 'experiment_08_testing_set.csv'
x_train, y_train = read_data(train_filename)
x_test, y_test = read_data(train_filename)

#开始训练
T_list = list(range(1, 11))
acc_list = []
h_list = []
for T in T_list:
    h_list, alpha_list = AdaBoost(x_train, y_train, T)
    y_pred = prediction(h_list, alpha_list, x_test)
    acc = np.sum(y_pred == y_test) / len(y_test)
    acc_list.append(acc)
    print('T =', T, 'Accuracy =%.2f ' % acc)

#绘图
plot_boundary(x_test, y_test, h_list, alpha_list)
plt.plot(T_list, acc_list, marker='o')
plt.xlabel('Number of Base Models T')
plt.ylabel('Accuracy on Test Set')
plt.xticks(T_list)
plt.show()