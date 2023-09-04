import pandas as pd
from sklearn import datasets
import numpy as np

train_data = pd.DataFrame(pd.read_csv('experiment_07_training_set.csv'))
x_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
test_data = pd.DataFrame(pd.read_csv('experiment_07_testing_set.csv'))
x_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

# 创建一个字典，将字符串颜色映射为数字值
type_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
# 使用 replace() 函数将字符串值用数字表示
y_train = y_train.replace(type_mapping)
y_test = y_test.replace(type_mapping)

# 转为dataframe
train = pd.DataFrame(x_train)
train.insert(0, 'target', y_train)
# 训练过程（算公式过程）
# 开辟一个3*8的ndarray存3类4个特征的均值和标准差
w = np.zeros((3, 8))
# 存储每类的数量
type_size = np.zeros(3)

# 分别对三类数据对应特征求均值和标准差
for type in range(3):
    temp = train[train['target'] == type]
    type_size[type] = len(temp)
    temp = temp.iloc[:, 1:]
    mean = temp.mean()
    std = temp.std()
    for feature in range(4):
        w[type][feature * 2] = mean.iloc[feature]
        w[type][feature * 2 + 1] = std.iloc[feature]


# 求正态概率密度
def normal_f(mean, std, x):
    x = np.array(x)
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.square(x - mean) /
                                                   (2 * np.square(std)))


former_prob = []
acc_num = 0
for type in range(3):
    #先验概率赋值给后验概率
    former_prob.append(type_size[type] / len(x_train))
for i in range(len(x_test)):
    # 后验概率
    post_prob = np.zeros(3)
    for type in range(3):
        #先验概率赋值给后验概率
        post_prob[type] = former_prob[type]

        for feature in range(4):
            post_prob[type] *= normal_f(w[type][feature * 2],
                                        w[type][feature * 2 + 1],
                                        x_test.iloc[i, feature])
    ans = np.argmax(post_prob)
    #print("guess:", ans, '\n', "right_ans:", y_test[i])
    if ans == y_test[i]:
        acc_num += 1

w = pd.DataFrame(
    w,
    index=["P(X|Y=setosa)", "P(X|Y=versicolor)", "P(X|Y=virginica)"],
    columns=[
        "X1=SepalLength_mean", "X1=SepalLength_std", "X2=SepalWidthCm_mean",
        "X2=SepalWidthCm_std", "X3=PetalLength_mean", "X3=PetalLength_std",
        "X4=PetalWidth_mean", "X4=PetalWidth_std"
    ])
former_prob = pd.DataFrame(
    former_prob,
    index=["P(X|Y=setosa)", "P(X|Y=versicolor)", "P(X|Y=virginica)"],
    columns=["先验概率"])
pd.set_option("display.max_columns", None)
print("acc:", acc_num / len(x_test))
print(former_prob)
print(w)
