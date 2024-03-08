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
from FacesDataset import FacesDataset
from torchvision.transforms import ToPILImage


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


if __name__=="__main__":
    # print(torch.cuda.is_available())
    to_pil = ToPILImage()

    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # get global configuration object
    compiler_cfg.balancer_policy = "Ribbon"  # set balancer policy
    compiler_cfg.enable_t_streaming = True  # enable tensor streaming

    tt0 = TTDevice("grayskull0")
    discriminator = pybuda.PyTorchModule('DiscriminatorV4', DiscriminatorV4())
    tt0.place_module(discriminator)

    # get an image from the image dataset
    training_set = FacesDataset('datasets/train.csv')
    item = training_set.__getitem__(0)
    image_tensor, label, image_path = item
    # print(training_set.__getitem__(0))
    # print(image_tensor)
    # print(image_tensor.shape)
    pil_image = to_pil(image_tensor)
    pil_image.save("test.png")

    act = torch.randn(1, 3, 256, 256)
    print(act.shape)

    # run a test forward pass on the net
    tt0.push_to_inputs(act)
    result = pybuda.run_inference(input_count=1, _sequential=True)
    val = result.get()[0]
    print(val.value())