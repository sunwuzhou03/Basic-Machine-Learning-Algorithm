import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def k_means(data, k, iterations=100):
    # 随机选取k个中心点
    centers = random.sample(list(data), k)
    for i in range(iterations):
        # 初始化簇
        clusters = [[] for _ in range(k)]
        # 将每个数据点分配给距离最近的中心点所在的簇
        for point in data:
            distances = [np.linalg.norm(point - center) for center in centers]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        # 更新中心点
        for j in range(k):
            centers[j] = np.mean(clusters[j], axis=0)
    # 返回簇和中心点
    return clusters, centers


def read_data(filename):
    data = pd.read_csv(filename).values.astype('float32')
    return data


# 定义数据集
filename = "experiment_10_training_set.csv"
data = read_data(filename)

# 绘制不同k值时的聚类结果图和loss值变化曲线图
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
plt.subplots_adjust(hspace=0.5)
loss_list = []  # 记录loss值
for i in range(10):
    row = i // 5
    col = i % 5
    ax[row][col].set_title("k={}".format(i + 1))
    clusters, _ = k_means(data, i + 1)
    loss = 0
    for j in range(i + 1):
        cluster = np.array(clusters[j])
        ax[row][col].scatter(cluster[:, 0],
                             cluster[:, 1],
                             color=np.random.rand(3))
        loss += np.sum((cluster - np.mean(cluster, axis=0))**2)
    loss_list.append(loss)
plt.show()

# 绘制loss值随k值增加的变化曲线图
plt.plot(range(1, 11), loss_list, "bo-")
plt.xlabel("k")
plt.ylabel("SSE")
plt.title("SSE vs. k")
plt.show()