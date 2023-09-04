import numpy as np
import pandas as pd
import sys

sys.path.append('模型文件')
from OVR逻辑回归 import read_data, data_normalization, OVRLogistic
from OVR支持向量机 import OVRSVM
from 神经网络 import Neural_Network

if __name__ == '__main__':
    name_list = [
        "iris", "digits", "wine", "breast_cancer",
        "data_banknote_authentication", "dermatology"
    ]
    result_filename = "模型测试"
    result_filename = f"{result_filename}//测试结果//result.txt"
    with open(result_filename, "w") as f:
        f.write(f"测试结果:\n")
    for dataset_name in name_list:
        dataset_filename = f"数据集//测试集//{dataset_name}.csv"
        params_filename = f"模型参数"

        label, data = read_data(dataset_filename)
        data = data_normalization(data)
        label_df = pd.DataFrame(label.copy())
        label = label.reshape(-1, 1)

        print(f"基于测试集{dataset_name}开始测试模型..")

        with open(result_filename, "a") as f:
            f.write(f"基于测试集{dataset_name}开始测试模型..\n")

        print("神经网络:", end=' ')
        input_dim = data.shape[1]
        hidden_dim = data.shape[1]
        output_dim = len(label_df.iloc[:, -1].unique())
        lr = 0.3
        batch_size = 4
        N = 500
        model = Neural_Network(input_dim, hidden_dim, output_dim, lr,
                               batch_size, N)
        model.load_params(f"{params_filename}//神经网络//{dataset_name}")
        results = model.predict(data)
        accuracy = (np.sum(results == label) / label.shape[0])
        print(round(accuracy, 2))
        with open(result_filename, "a") as f:
            f.write(f"神经网络: {round(accuracy,2)}\n")

        print("OVR逻辑回归:", end=' ')
        alpha = 1
        N = 500
        model = OVRLogistic(input_dim, output_dim, alpha, N)
        model.load_params(f"{params_filename}//OVR逻辑回归//{dataset_name}")
        results = model.predict(data)
        accuracy = (np.sum(results == label) / label.shape[0])
        print(round(accuracy, 2))
        with open(result_filename, "a") as f:
            f.write(f"OVR逻辑回归: {round(accuracy,2)}\n")

        print("OVR支持向量机:", end=' ')
        C = 100
        alpha = 0.0001
        N = 500
        model = OVRSVM(input_dim, output_dim, C, alpha, N)
        model.load_params(f"{params_filename}//OVR支持向量机//{dataset_name}")
        results = model.predict(data)
        accuracy = (np.sum(results == label) / label.shape[0])
        print(round(accuracy, 2))
        with open(result_filename, "a") as f:
            f.write(f"OVR支持向量机: {round(accuracy,2)}\n\n")
        print("\n")