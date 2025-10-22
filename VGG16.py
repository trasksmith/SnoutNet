import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from SnoutNetData import SnoutNetData  # your custom dataset

class VGG16(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, output_dim)

    def forward(self, x):
        return self.backbone(x)