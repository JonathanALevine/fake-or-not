#!/usr/bin/env python3
import torch
from torch import nn
torch.manual_seed(18)
torch.cuda.is_available()
from datetime import datetime
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class InverseMaxPool2d(nn.Module):
    def __init__(self, scale_factor):
        super(InverseMaxPool2d, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        # Nearest neighbor interpolation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        # Convolution with learnable filter
        x = nn.Conv2d(in_channels=x.shape[1], out_channels=x.shape[1], kernel_size=3, stride=1, padding=1)(x)
        return x


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(32, 131072),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(View(32, 64, 64),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    InverseMaxPool2d(scale_factor=1),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    InverseMaxPool2d(scale_factor=1),
                            nn.ReLU())
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                                    InverseMaxPool2d(scale_factor=1),
                                    nn.ReLU())


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x        


def main():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()
    test_tensor = torch.rand(1, 32)
    print(test_tensor, test_tensor.shape)
    model = Generator()
    image_tensor = model.forward(test_tensor)
    print(image_tensor.shape)
    image = transforms.ToPILImage()(image_tensor.squeeze(0))
    image.show()
    