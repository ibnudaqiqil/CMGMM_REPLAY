import torch
import torch.nn as nn
import torch.nn.functional as F
# A basic feedforward net
class FNet(nn.Module):
  def __init__(self, hsize=512):
    super(FNet, self).__init__()

    self.l1 = nn.Linear(784, hsize)
    self.l2 = nn.Linear(hsize, 10)

  def forward(self, x):
      x = x.view(x.size(0), -1)
      x = F.relu(self.l1(x))
      x = self.l2(x)
      return x


# A basic feedforward net

class FNetCNN(nn.Module):
    def __init__(self):
        super(FNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
