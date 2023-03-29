#!/usr/bin/env python3
import torch
from torch import nn
torch.manual_seed(18)
torch.cuda.is_available()
from datetime import datetime


def main():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()