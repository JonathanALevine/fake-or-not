#!/usr/bin/env python3
import pybuda
import pybuda.op
from pybuda.tensor import TensorFromPytorch
import torch
from torch import nn
from pybuda import PyBudaModule, TTDevice
from pybuda import CPUDevice
from pybuda.op.convolution import Conv2d
from pybuda import PyTorchModule


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
        # self.layer3 = nn.Sequential(ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=0))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(in_features=1152, out_features=256),
                                    nn.ReLU(),
                                    nn.Linear(in_features=256, out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=32),
                                    nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=1),
                                    nn.Sigmoid())
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__=="__main__":
    print(torch.cuda.is_available())
    tt0 = TTDevice("grayskull0")
    discriminator = pybuda.PyTorchModule('DiscriminatorV4', DiscriminatorV4())
    # weight = torch.rand(*act_dim).half()
    # discriminator.set_parameter("weights",  weight)
    tt0.place_module(discriminator)

    act = torch.randn(1, 3, 128, 128)

    # run a test forward pass on the net
    tt0.push_to_inputs(act)
    result = pybuda.run_inference(input_count=1, _sequential=True)
    val = result.get()[0]
    print(val.value())
    print(val.shape)

    # Add a CPU device to calculate loss
    params = discriminator.parameters()
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    # cpu = CPUDevice("cpu0", optimizer_f = None, scheduler_f = None)
    cpu = CPUDevice("cpu0", optimizer_f=optimizer, scheduler_f=None)
    # declare the loss function
    loss_module = PyTorchModule("bceloss", nn.BCELoss())
    cpu.place_loss_module(loss_module)

    target = torch.tensor([1])
    cpu.push_to_target_inputs(target)
    print(target)

    tt0.push_to_inputs(act)
    pybuda.run_training(epochs=1, steps=1, accumulation_steps=1, microbatch_count=1)