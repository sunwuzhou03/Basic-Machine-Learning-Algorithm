import numpy as np
import pandas as pd
import os
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
    for dataset_name in name_list:
        dataset_filename = f"数据集//训练集//{dataset_name}.csv"

        #读取数据并调整数据形状
        label, data = read_data(dataset_filename)
        data = data_normalization(data)
        label_df = pd.DataFrame(label.copy())
        label = label.reshape(-1, 1)

        print(f"基于完整训练集{dataset_name}开始训练模型..")

        print("神经网络...")
        param_filename = f"模型参数//神经网络//{dataset_name}"
        input_dim = data.shape[1]
        hidden_dim = data.shape[1]
        output_dim = len(label_df.iloc[:, -1].unique())
        lr = 0.3
        batch_size = 4
        N = 500
        model = Neural_Network(input_dim, hidden_dim, output_dim, lr,
                               batch_size, N)
        model.fit(data, label)
        model.save_params(param_filename)

        print("OVR逻辑回归..")
        param_filename = f"模型参数//OVR逻辑回归//{dataset_name}"
        alpha = 1
        N = 500
        model = OVRLogistic(input_dim, output_dim, alpha, N)
        model.fit(data, label)
        model.save_params(param_filename)

        print("OVR支持向量机..")
        param_filename = f"模型参数//OVR支持向量机//{dataset_name}"
        C = 100
        alpha = 0.0001
        N = 500
        model = OVRSVM(input_dim, output_dim, C, alpha, N)
        model.fit(data, label)
        model.save_params(param_filename)

        print("\n")