import pandas as pd
import sklearn.datasets as datasets
import numpy as np

# 加载数据集
iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()

# 读取csv格式数据
data_banknote_authentication_df = pd.DataFrame(
    pd.read_csv("./数据集//原始数据集/data_banknote_authentication.txt",
                sep=',',
                header=None), )
dermatology_df = pd.DataFrame(
    pd.read_csv("./数据集//原始数据集/archive/dermatology_database_1.csv"))

#数据清洗
dermatology_df['age'] = dermatology_df['age'].replace('?', np.nan)
dermatology_df.dropna(subset=['age'], inplace=True)
dermatology_df['class'] = dermatology_df['class'].values.astype(float) - 1

# 将数据集转换为dataframe格式，同时保存标签
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
digits_df = pd.DataFrame(data=digits.data)
digits_df['target'] = digits.target
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
breast_cancer_df = pd.DataFrame(data=breast_cancer.data,
                                columns=breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target

# 将dataframe保存为csv文件

iris_df.to_csv('./数据集//原始数据集/iris.csv', index=False)
digits_df.to_csv('./数据集//原始数据集/digits.csv', index=False)
wine_df.to_csv('./数据集//原始数据集/wine.csv', index=False)
breast_cancer_df.to_csv('./数据集//原始数据集/breast_cancer.csv', index=False)
data_banknote_authentication_df.to_csv(
    './数据集//原始数据集/data_banknote_authentication.csv', index=False)
dermatology_df.to_csv('./数据集//原始数据集/dermatology.csv', index=False)