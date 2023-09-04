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

sys.path.append('model_adjust')
from 卷积神经网络调整 import CNN, MyDataset, read_data

device = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 10

data = read_data('data//MNIST//raw//mnist_train.csv')
allData = MyDataset(data)

allDataLoader = torch.utils.data.DataLoader(dataset=allData,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

optimizers = [
    "ASGD", "SGD", "Adagrad", "Adadelta", "RMSprop", "Adamax", "Adam", "Rprop"
]
choices = [1]
for choice in choices:

    lr = [0.001, 0.01, 0.05, 0.1]
    for learning_rate in lr:
        fig, ax = plt.subplots()

        ax.set_title(f"model_loss_curve")
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss per epoch")
        fig, ax = plt.subplots()

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

            total_loss = []
            per_loss = 0
            correct = 0
            accuracy_list = []
            for epoch in range(1, EPOCHS + 1):
                processBar = tqdm(allDataLoader, unit='step')
                net.train(True)
                for step, (trainImgs, labels) in enumerate(processBar):
                    trainImgs = trainImgs.to(device)
                    labels = labels.to(device)
                    net.zero_grad()
                    outputs = net(trainImgs)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = torch.sum(
                        predictions == labels) / labels.shape[0]

                    correct += torch.sum(predictions == labels)
                    per_loss = per_loss + loss.item()

                    loss.backward()

                    optimizer.step()
                    processBar.set_description(
                        "[%d/%d] choice: %d Loss: %.4f, Acc: %.4f" %
                        (epoch, EPOCHS, choice, loss.item(), accuracy.item()))

                per_loss = per_loss / len(allDataLoader)
                total_loss.append(per_loss)
                per_loss = 0

                accuracy = correct / (BATCH_SIZE * len(allDataLoader))
                accuracy_list.append(accuracy.item())
                correct = 0

            path = f"save_model//CNN//{learning_rate}"
            model_path = os.makedirs(path, exist_ok=True)
            model_path = path
            torch.save(net.state_dict(),
                       f"{model_path}//{optimizer_name}_cnn.pth")

            index = np.arange(1, EPOCHS + 1, 1)
            if choice:
                plt.plot(index, total_loss, label=f"{optimizer_name}")
                font = {
                    'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 10
                }
                plt.xticks(fontproperties='Times New Roman', fontsize=10)
                plt.xticks(fontproperties='Times New Roman', fontsize=10)
                plt.xlabel(u'epochs')
                plt.ylabel(u'loss')
                plt.legend(loc=1, prop=font)
            else:
                plt.plot(index, accuracy_list, label=f"{optimizer_name}")
                font = {
                    'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 10
                }
                plt.xticks(fontproperties='Times New Roman', fontsize=10)
                plt.xticks(fontproperties='Times New Roman', fontsize=10)
                plt.xlabel(u'epoch')
                plt.ylabel(u'accuracy')
                plt.legend(loc=1, prop=font)

        if choice:
            plt.title(f"{learning_rate}_loss_curve")
            plt.savefig(f"loss_curve//{learning_rate}_loss_curve.png")
            plt.show(block=False)
        else:
            plt.title(f"{learning_rate}_accuracy_curve")
            plt.savefig(f"accuracy_curve//{learning_rate}_accuracy_curve.png")
            plt.show(block=False)
