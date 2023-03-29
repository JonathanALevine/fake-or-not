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


class MyDataset(torch.utils.data.Dataset):
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
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class FaceNetV3(nn.Module):
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
    

    def save(self):
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        model_path = f"models/face-net-{date_string}.pth"
        torch.save(self.state_dict(), model_path)
    

@torch.no_grad()
def get_test_acc(model, test_loader, device='cuda'):
    model.to(device)
    metric = BinaryAccuracy()
    metric.to(device)
    test_acc = 0.0
    model.eval()
    for batch in tqdm(test_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        metric.update(outputs.flatten(), targets)
        test_acc += metric.compute().item()
    return test_acc / test_loader.__len__()


@torch.no_grad()
def estimate_performance(model, data_loader, device='cuda'):
    loss_value = 0.0
    acc_value = 0.0
    loss_fn = nn.BCELoss()
    loss_fn.to(device)
    metric = BinaryAccuracy()
    metric.to(device)
    model.eval()
    for batch in tqdm(data_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs.flatten().float(), targets.float())
        metric.update(outputs.flatten(), targets)
        loss_value += loss.item()
        acc_value += metric.compute().item()
    return loss_value, acc_value


def train(model, optimizer, train_loader, val_loader, epochs, loss_fn, device='cuda'):
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
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.flatten().float(), targets.float())
            loss.backward()
            optimizer.step()
            metric.update(outputs.flatten(), targets)
            train_loss += loss.item()
            train_acc += metric.compute().item()

        # evaluate the model on the validation set
        val_loss, val_acc = estimate_performance(model, val_loader, device)

        print(epoch_system_out_string(epoch, train_loss, train_acc, val_loss, val_acc, train_loader, val_loader))


def epoch_system_out_string(epoch:int, train_loss:float, train_acc:float, val_loss:float, val_acc:float, train_loader, val_loader)->str:
    output_string = f'''Epoch: {epoch} -- train Loss: {round(train_loss / train_loader.__len__(), 4)} 
                    valid Loss: {round(val_loss / val_loader.__len__(), 4)} 
                    train acc.:{round(train_acc / train_loader.__len__(), 4)} 
                    val acc.:{round(val_acc / val_loader.__len__(), 4)}'''
    return output_string


def main():
    batch_size = 128

    training_set = MyDataset('datasets/train.csv')
    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

    valid_set = MyDataset('datasets/valid.csv')
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

    test_set = MyDataset('datasets/test.csv')
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    model = FaceNetV3()

    loss_fn = nn.BCELoss()

    params = model.parameters()
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # train the model
    num_epochs = 1
    train(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=valid_loader, loss_fn=loss_fn, epochs=num_epochs)

    print(get_test_acc(model, test_loader))

    model.save()

if __name__ == "__main__":
    main()