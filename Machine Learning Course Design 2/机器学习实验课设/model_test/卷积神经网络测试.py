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
allData = MyDataset(data)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = len(data)
EPOCHS = 10

allDataLoader = torch.utils.data.DataLoader(dataset=allData,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

optimizers = [
    "ASGD", "SGD", "Adagrad", "Adadelta", "RMSprop", "Adamax", "Adam", "Rprop"
]
learning_rate = 0.001
optimizer_name = "Adam"
net = CNN()
net.to(device)
print(f"optimizer_name={optimizer_name}, learning_rate={learning_rate}! ")
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
        print(
            classification_report(labels.cpu().numpy(),
                                  predictions.cpu().numpy(),
                                  digits=3))
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        print(accuracy)