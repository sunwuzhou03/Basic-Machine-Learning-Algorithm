import numpy as np
import pandas as pd
import os
import sys

sys.path.append('模型文件')
from OVR逻辑回归 import read_data, data_normalization, OVRLogistic
from OVR支持向量机 import OVRSVM
from 神经网络 import Neural_Network


#返回accuracy最高的模型
def k_fold_cross_validation(
    model,
    X,
    y,
    result_filename,
    k=5,
    random_state=42,
):
    # 对索引数组进行随机洗牌
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    #数据集随机化
    X = X[indices]
    y = y[indices]

    # 将数据均分为k个段
    subset_size = len(X) // k
    subsets_X = [X[i:i + subset_size] for i in range(0, len(X), subset_size)]
    subsets_y = [y[i:i + subset_size] for i in range(0, len(y), subset_size)]

    #寻找最优准确率和最优模型
    best_accuracy = 0
    worst_accuracy = 1
    average_accuracy = 0
    best_model = None

    # 进行k次循环，每一次取出第k份为验证集，其余为训练集
    for i in range(k):
        X_train, y_train = [], []
        for j in range(k):
            if j != i:
                X_train.append(subsets_X[j])
                y_train.append(subsets_y[j])

        X_train = np.array(np.concatenate(X_train))
        X_test, y_test = subsets_X[i], subsets_y[i]

        y_test = y_test.reshape(-1, 1)
        y_train = np.concatenate(y_train).reshape(-1, 1)
        model.fit(X_train, y_train)

        results = model.predict(X_test)
        accuracy = (np.sum(results == y_test) / y_test.shape[0])

        steps = 10
        #寻找最优模型
        worst_accuracy = min(accuracy, worst_accuracy)
        average_accuracy += (accuracy - average_accuracy) / (i + 1)
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
        print(f"step={i},accuracy={round(accuracy,2)}")

        #运行结果写入文件
        with open(result_filename, "a") as f:
            f.write(f"step={i},accuracy={round(accuracy,2)}\n")

        #模型重置
        model.reset()
    #返回最优模型

    with open(result_filename, "a") as f:
        f.write(
            f"best_accuracy={round(best_accuracy,2)}\nworst_accuracy={round(worst_accuracy,2)}\naverage_accuracy={round(average_accuracy,2)}\n\n"
        )

    print()


if __name__ == '__main__':
    name_list = [
        "iris", "digits", "wine", "breast_cancer",
        "data_banknote_authentication", "dermatology"
    ]
    for dataset_name in name_list:
        print(f"基于数据集{dataset_name}的验证")
        dataset_filename = f"数据集//训练集//{dataset_name}.csv"
        result_filename = f"k折交叉验证(调参)\\交叉验证结果"

        label, data = read_data(dataset_filename)
        data = data_normalization(data)
        label_df = pd.DataFrame(label.copy())
        label = label.reshape(-1, 1)

        result_path = os.makedirs(f"{result_filename}", exist_ok=True)
        result_path = result_filename
        result_filename = f"{result_filename}\{dataset_name}.txt"

        with open(result_filename, "w") as f:
            f.write(f"神经网络:\n")

        print("神经网络:")
        param_filename = f"模型参数//神经网络//{dataset_name}"
        input_dim = data.shape[1]
        hidden_dim = data.shape[1]
        output_dim = len(label_df.iloc[:, -1].unique())
        lr = 0.3
        batch_size = 4
        N = 500
        model = Neural_Network(input_dim, hidden_dim, output_dim, lr,
                               batch_size, N)
        k_fold_cross_validation(
            model,
            data,
            label,
            result_filename,
            10,
        )
        model.fit(data, label)
        model.save_params(param_filename)

        with open(result_filename, "a") as f:
            f.write(f"OVR逻辑回归:\n")

        print("OVR逻辑回归:")
        param_filename = f"模型参数//OVR逻辑回归//{dataset_name}"
        alpha = 1
        N = 500
        model = OVRLogistic(input_dim, output_dim, alpha, N)
        k_fold_cross_validation(model, data, label, result_filename, 10)
        model.fit(data, label)
        model.save_params(param_filename)

        with open(result_filename, "a") as f:
            f.write(f"OVR支持向量机:\n")

        print("OVR支持向量机:")
        param_filename = f"模型参数//OVR支持向量机//{dataset_name}"
        C = 100
        alpha = 0.0001
        N = 500
        model = OVRSVM(input_dim, output_dim, C, alpha, N)
        k_fold_cross_validation(model, data, label, result_filename, 10)
        model.fit(data, label)
        model.save_params(param_filename)
