import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import sys
import os
from sklearn.metrics import classification_report

sys.path.append('model_adjust')
from 卷积神经网络调整 import CNN, MyDataset, read_data

data = read_data('data//MNIST//raw//mnist_test.csv')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = len(data)
EPOCHS = 10

allData = MyDataset(data)

allDataLoader = torch.utils.data.DataLoader(dataset=allData,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

optimizers = [
    "ASGD", "SGD", "Adagrad", "Adadelta", "RMSprop", "Adamax", "Adam", "Rprop"
]
lr = [0.001, 0.01, 0.05, 0.1]
for learning_rate in lr:
    accuracy_list = []

    with open(f"model_test//CNN//{learning_rate}_Eva.txt", "w") as f:
        f.write(f"learning_rate = {learning_rate} model evaluate\n")

    for optimizer_name in optimizers:
        net = CNN()
        net.to(device)
        print(
            f"optimizer_name={optimizer_name}, learning_rate={learning_rate}! "
        )
        lossF = torch.nn.CrossEntropyLoss()
        optimizers = {
            "ASGD": torch.optim.ASGD,
            "SGD": torch.optim.SGD,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta,
            "RMSprop": torch.optim.RMSprop,
            "Adamax": torch.optim.Adamax,
            "Adam": torch.optim.Adam,
            "Rprop": torch.optim.Rprop,
        }

        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        optimizer_class = optimizers[optimizer_name]
        optimizer = optimizer_class(net.parameters(), lr=learning_rate)

        correct = 0
        net.train(False)

        path = f"save_model//CNN//{learning_rate}"
        model_path = os.makedirs(path, exist_ok=True)
        model_path = path
        net.load_state_dict(
            torch.load(f"{model_path}//{optimizer_name}_cnn.pth"))  # 同上

        correct, totalLoss = 0, 0
        net.train(False)

        with torch.no_grad():
            for step, (testImgs, labels) in enumerate(allDataLoader):
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                net.zero_grad()
                outputs = net(testImgs)
                loss = lossF(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum(predictions == labels) / labels.shape[0]

                correct += torch.sum(predictions == labels)

                with open(f"model_test//CNN//{learning_rate}_Eva.txt",
                          "a") as f:
                    f.write(
                        f"optimizer = {optimizer_name}\n{classification_report(labels.cpu().numpy(),predictions.cpu().numpy(),digits=2)}\n"
                    )

                    accuracy = correct / (BATCH_SIZE * len(allDataLoader))
                    accuracy_list.append(accuracy.item())
                    print(accuracy.item())
                    # print((BATCH_SIZE * len(allDataLoader)))

    x = np.arange(1, 9)
    y = accuracy_list
    fig = plt.figure(figsize=(8, 6))
    plt.bar(x, y)
    plt.xticks(x, optimizers, rotation=-45)
    plt.xlabel('optimizer')
    plt.ylabel('accuracy')
    plt.title(f"{learning_rate}_Accuracy comparison")

    for i in range(len(x)):
        plt.text(x[i], y[i], '%.2f' % round(y[i], 2), ha='center', va='bottom')

    plt.savefig(f"model_test//CNN//{learning_rate}_AccCom.png")
    plt.show()