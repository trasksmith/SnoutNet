import torch
import torch.nn as nn
import torch.nn.functional as F

class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(4*4*256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(self.pool(x))

        # Conv block 2
        x = self.conv2(x)
        x = F.relu(self.pool(x))

        # Conv block 3
        x = self.conv3(x)
        x = F.relu(self.pool(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x