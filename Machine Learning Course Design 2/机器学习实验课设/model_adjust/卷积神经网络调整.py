import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = torch.nn.Sequential(
            #The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1))

    #前向传播
    def forward(self, input):
        output = self.model(input)
        return output


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0]  # 第一列为标签
        data = self.data[idx, 1:]  # 从第2列开始作为数据
        data = data.reshape((1, 28, 28))
        return data / 255.0, int(label)


#读取数据函数，返回属性和标签
def read_data(filename):
    data = pd.read_csv(filename, header=None).values.astype('float32')
    return data


def split_dataset(dataset, ratio, random_state=42):
    dataset_copy = np.array(dataset).copy()
    train_size = int(len(dataset_copy) * ratio)
    train_set = []
    valid_set = list(dataset_copy)
    np.random.seed(random_state)
    while len(train_set) < train_size:
        index = random.randrange(len(valid_set))
        train_set.append(valid_set.pop(index))
    return np.array(train_set), np.array(valid_set)


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    EPOCHS = 10
    learning_rate = 0.001  #0.001，0.01，0.1

    # trainData = MyDataset('data//MNIST//raw//mnist_train.csv')
    # testData = MyDataset('data//MNIST//raw//mnist_test.csv')

    data = read_data('data//MNIST//raw//mnist_train.csv')

    trainData, validData = split_dataset(data, 0.8)
    trainData = MyDataset(trainData)
    validData = MyDataset(validData)

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)

    testDataLoader = torch.utils.data.DataLoader(dataset=validData,
                                                 batch_size=BATCH_SIZE)

    net = CNN()
    print(net.to(device))

    lossF = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Rprop(net.parameters(), lr=learning_rate)

    history = {'Test Loss': [], 'Test Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(trainDataLoader, unit='step')
        net.train(True)
        for step, (trainImgs, labels) in enumerate(processBar):
            print(trainImgs[trainImgs > 1])
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)
            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]
            loss.backward()

            optimizer.step()
            processBar.set_description(
                "[%d/%d] Loss: %.4f, Acc: %.4f" %
                (epoch, EPOCHS, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                with torch.no_grad():
                    for testImgs, labels in testDataLoader:
                        testImgs = testImgs.to(device)
                        labels = labels.to(device)
                        outputs = net(testImgs)
                        loss = lossF(outputs, labels)
                        predictions = torch.argmax(outputs, dim=1)

                        totalLoss += loss
                        correct += torch.sum(predictions == labels)

                    testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                    testLoss = totalLoss / len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())

                processBar.set_description(
                    "[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f"
                    % (epoch, EPOCHS, loss.item(), accuracy.item(),
                       testLoss.item(), testAccuracy.item()))

        processBar.close()
