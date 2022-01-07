import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from datasets.UrbanSoundDataset import UrbanSoundDataset
import torchaudio
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

csv_path = './data/UrbanSound8K/metadata/UrbanSound8K.csv'
file_path = './data/UrbanSound8K/audio/'

train_set = UrbanSoundDataset(csv_path, file_path, range(1, 10))
test_set = UrbanSoundDataset(csv_path, file_path, [10])
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

# needed for using datasets on gpu
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        # input should be 512x30 so this outputs a 512x1
        self.avgPool = nn.AvgPool1d(30)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = Net()
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_()  # set requires_grad to True for training
        output = model(data)
        # original output dimensions are batchSizex1x10
        output = output.permute(1, 0, 2)
        # the loss functions expects a batchSizex10 input
        loss = F.nll_loss(output[0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(model, test_loader,  optimizer, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



for epoch in range(1, 50):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    train(model, train_loader, optimizer,epoch)
    test(model, test_loader, optimizer, epoch)
