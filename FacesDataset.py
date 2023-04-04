#!/usr/bin/env python3
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
torch.manual_seed(18)


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path):
        self.data = pd.read_csv(data_file_path)
        self.image_transform = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, idx):
        image_path = self.data['path'][idx]
        image = Image.open(image_path)
        label = self.data['label'][idx]
        return self.image_transform(image), label, image_path
    

    def __len__(self):
        return len(self.data)