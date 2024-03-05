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
from DiscriminatorV4 import DiscriminatorV4, ConvBlock
from FacesDataset import FacesDataset
import matplotlib.pyplot as plt
import pybuda


__DEVICE__ = 'cpu'


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
    # output = pybuda.PyTorchModule("direct_pt", DiscriminatorV4()).run(torch.randn(1, 3, 256, 256))

    # # try the forward pass on the cpu
    # # input = torch.randn(1, 3, 256, 256).to(__DEVICE__)
    # input = torch.rand(1, 10000)
    # output = model.forward(input)
    # print("Pytorch pass ......")
    # print(output)

    # try the forward pass on the tt-card
    # module = pybuda.PyTorchModule("direct_pt", myNet())

    # tt0 = pybuda.TTDevice("tt0", 
    #                       module=module, 
    #                       arch=pybuda.BackendDevice.Grayskull,
    #                       devtype=pybuda.BackendType.Silicon)

    # tt0 = pybuda.TTDevice("grayskull0")
    # tt0.place_module(module)
    
    # for i in range(1000):
    #     input = torch.rand(1, 10000)
    #     # print(input)
    #     print(f"PyBUDA forward pass ..... {i}")
    #     output = pybuda.run_inference(inputs=[input])
    #     print(output.get())

    # input = torch.rand(1, 1)
    # output = pybuda.run_inference(inputs=[input])
    # # output = pybuda.PyTorchModule("direct_pt", myNet()).run(input)
    # print("PyBUDA forward pass .....")
    # print(output)

    # # now try the CNN on the cpu
    # model = DiscriminatorV4()
    # input = torch.randn(1, 3, 256, 256)
    # output = model.forward(input)
    # print("PyTorch forward pass ......")
    # print(output)

    # # try the forward pass on the tt-card
    # input = torch.randn(1, 3, 256, 256)
    # output = pybuda.PyTorchModule("direct_pt", DiscriminatorV4()).run(input)
    # print("PyBUDA forward pass .....")
    # print(output)

    # try the forward pass on the tt-card
    # input = torch.rand(1, 1)
    # output = pybuda.PyTorchModule("direct_pt", myNet()).run(input)
    # print("PyBUDA forward pass .....")
    # print(output)

    loss_fn = nn.BCELoss()

    params = model.parameters()
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # train the model
    num_epochs = 1
    train(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=valid_loader, test_loader=test_loader,loss_fn=loss_fn, epochs=num_epochs)