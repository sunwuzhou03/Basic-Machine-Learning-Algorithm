import pandas as pd
import random
import numpy as np
import os


def split_dataset(dataset, ratio, random_state=42):
    train_size = int(len(dataset) * ratio)
    train_set = []
    test_set = list(dataset)
    np.random.seed(random_state)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return pd.DataFrame(train_set), pd.DataFrame(test_set)


def split_data(filename, dataset_name, ratio=0.2):
    """
    读取CSV文件并将其分为训练集和测试集，然后将两个数据集保存为新的CSV文件
    :param filename: CSV文件名
    :param test_size: 测试集所占比例，默认为0.2
    :return: 无返回值
    """
    # 读取CSV文件到pandas的DataFrame对象
    data_df = pd.read_csv(filename).values.astype(float)

    # 将数据集划分为训练集和测试集，test_size表示测试集所占比例
    train_df, test_df = split_dataset(data_df, ratio=ratio)

    # 将训练集和测试集保存为CSV文件
    os.makedirs(f"数据集//训练集", exist_ok=True)
    os.makedirs(f"数据集//测试集", exist_ok=True)
    train_df.to_csv(f"数据集//训练集//{dataset_name}.csv", index=False)
    test_df.to_csv(f"数据集//测试集//{dataset_name}.csv", index=False)


if __name__ == '__main__':
    name_list = [
        "breast_cancer", "data_banknote_authentication", "dermatology",
        "digits", "iris", "wine"
    ]
    for dataset_name in name_list:
        dataset_filename = f"数据集//原始数据集//{dataset_name}.csv"
        split_data(dataset_filename, dataset_name, 0.7)
