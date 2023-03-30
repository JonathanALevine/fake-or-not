#!/usr/bin/env python3
import torch
from torch import nn
torch.manual_seed(18)
torch.cuda.is_available()
from datetime import datetime
from PIL import Image
from torchvision import transforms


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        pass


    def forward(self, x):
        pass


def main():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()
    test_tensor = torch.rand(3, 256, 256)
    print(test_tensor)
    image = transforms.ToPILImage()(test_tensor)
    image.show()
    