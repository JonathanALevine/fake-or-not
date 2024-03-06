#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics.functional import binary_accuracy
torch.manual_seed(18)
torch.cuda.is_available()
from DiscriminatorV3 import DiscriminatorV3, ConvBlock
from FacesDataset import FacesDataset
import matplotlib.pyplot as plt
import pybuda


__DEVICE__ = 'cpu'


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DiscriminatorV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(ConvBlock(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=2, padding=0))
        self.layer2 = nn.Sequential(ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=2, padding=0))
        self.layer3 = nn.Sequential(ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=0))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(in_features=128, out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=32),
                                    nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=16),
                                    nn.ReLU(),
                                    nn.Linear(in_features=16, out_features=1),
                                    nn.Sigmoid())
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def epoch_system_out_string(epoch:int, train_loss:float, train_acc:float, val_loss:float, val_acc:float, test_acc:float)->str:
    return (f'Epoch: {epoch} -- train Loss: {round(train_loss, 4)} \t valid Loss: {round(val_loss, 4)} \t train acc.:{round(train_acc, 4)} \t val acc.:{round(val_acc, 4)} \t test acc.:{round(test_acc, 4)}')


@torch.no_grad()
def estimate_performance(model, data_loader, device=__DEVICE__):
    loss_value = 0.0
    acc_value = 0.0
    loss_fn = nn.BCELoss()
    loss_fn.to(device)
    metric = BinaryAccuracy()
    metric.to(device)
    model.eval()
    for batch in tqdm(data_loader):
        inputs, targets, paths = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs.flatten().float(), targets.float())
        metric.update(outputs.flatten(), targets)
        loss_value += loss.item()
        acc_value += metric.compute().item()
    return loss_value/data_loader.__len__(), acc_value/data_loader.__len__()


def train(model, optimizer, train_loader, val_loader, test_loader, epochs, loss_fn, device=__DEVICE__):
    model.to(device)
    loss_fn.to(device)
    metric = BinaryAccuracy()
    metric.to(device)

    for epoch in (range(epochs)):
        # train the model on the training set
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs, targets, paths = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.flatten().float(), targets.float())
            loss.backward()
            optimizer.step()
            metric.update(outputs.flatten(), targets)
            train_loss += loss.item()
            train_acc += metric.compute().item()

        # estimate the performance of the model on the validation set and the test set
        train_loss, train_acc = train_loss / len(train_loader), train_acc / len(train_loader)
        val_loss, val_acc = estimate_performance(model, val_loader, device)
        test_loss, test_acc = estimate_performance(model, test_loader, device)

        print(epoch_system_out_string(epoch, train_loss, train_acc, val_loss, val_acc, test_acc))


class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10000, 3)
        self.output = nn.Linear(3, 1)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


if __name__=="__main__":
    print(torch.cuda.is_available())

    # set the batch size
    batch_size = 32

    # Make the train, valid and test data loaders
    training_set = FacesDataset('datasets/train.csv')
    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

    valid_set = FacesDataset('datasets/valid.csv')
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

    test_set = FacesDataset('datasets/test.csv')
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # Instatiate the model
    model = DiscriminatorV4().to(__DEVICE__)

    # test a forward pass on the discriminator
    input = torch.randn(1, 3, 256, 256).to(__DEVICE__)
    print(model.forward(input), model.forward(input).size())

    loss_fn = nn.BCELoss()

    params = model.parameters()
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # train the model
    num_epochs = 1
    train(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=valid_loader, test_loader=test_loader,loss_fn=loss_fn, epochs=num_epochs)