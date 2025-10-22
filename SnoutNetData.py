import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random

class SnoutNetData(Dataset):
    def __init__(self, path, labels_file, transform=None, geo_transform=False):
        self.path = path
        self.transform = transform
        self.geo_transform = geo_transform

        with open(labels_file, 'r') as f:
            lines = f.readlines()

        self.labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            filename, coord_str = line.split(',', 1)
            filename = filename.strip()
            coord_str = coord_str.strip().strip('"').strip('()')
            x, y = map(int, coord_str.split(','))
            self.labels.append((filename, (x, y)))

    def __len__(self):
        return len(self.labels)
    
    def _geometric_transform(self, image, coords):
        x, y = coords
        orig_w, orig_h = image.size

        # Resize to 227x227
        image = F.resize(image, (227, 227))
        x *= 227 / orig_w
        y *= 227 / orig_h

        # Random crop to 216x216
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(216, 216))
        image = F.crop(image, i, j, h, w)
        x -= j
        y -= i

        # Random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
            x = w - x

        # Resize back to 227x227
        x *= 227 / w
        y *= 227 / h

        return image, torch.tensor([x, y], dtype=torch.float32)

    def __getitem__(self, idx):
        filename, (x, y) = self.labels[idx]
        img_path = os.path.join(self.path, filename)
        image = Image.open(img_path).convert('RGB')
        coords = [x, y]

        orig_w, orig_h = image.size  # original size            

        if self.geo_transform:
            image, label = self._geometric_transform(image, coords)
            # label is already a tensor, scaled to 227x227 space
            # Optionally normalize label to [0,1] range for training
            label = label / 227.0
            # Apply color transforms if any (usually your self.transform should exclude normalization and ToTensor if geo_transform is True)
            if self.transform:
                image = self.transform(image)
        else:
            # No geometric transform: apply your standard transform (resize, color jitter, normalize, etc.)
            if self.transform:
                image = self.transform(image)
            label = torch.tensor([x / orig_w, y / orig_h], dtype=torch.float32)

        return image, label