#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics.functional import binary_accuracy
torch.manual_seed(18)
torch.cuda.is_available()


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path):
        self.data = pd.read_csv(data_file_path)
        self.image_transform = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, idx):
        image_path = self.data['path'][idx]
        image = Image.open(image_path)
        label = self.data['label'][idx]
        return self.image_transform(image), label
    

    def __len__(self):
        return len(self.data)