import torch
from torch import nn
torch.manual_seed(18)
torch.cuda.is_available()


class FaceNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(4, 4), stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=0),
                                    nn.Dropout2d(p=0.2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0),
                                    nn.Dropout2d(p=0.2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(6, 6), stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.AvgPool2d(kernel_size=(5, 5), stride=1, padding=0),
                                    nn.Dropout2d(p=0.2))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7, 7), stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.AvgPool2d(kernel_size=(6, 6), stride=1, padding=0),
                                    nn.Dropout2d(p=0.2))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(in_features=50176, out_features=32),
                                    nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=16),
                                    nn.ReLU(),
                                    nn.Linear(in_features=16, out_features=8),
                                    nn.ReLU(),
                                    nn.Linear(in_features=8, out_features=1),
                                    nn.Sigmoid())
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x