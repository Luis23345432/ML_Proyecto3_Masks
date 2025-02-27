import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random

class SkinLesionDataset(Dataset):
    def __init__(self, image_path, mask_path, augment=False):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augment = augment

        self.images = sorted([f for f in os.listdir(image_path) if f.endswith(".npy")])
        self.masks = sorted([f for f in os.listdir(mask_path) if f.endswith(".npy")])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_path, self.images[idx]))
        mask = np.load(os.path.join(self.mask_path, self.masks[idx]))

        mask = mask / 255.0

        if self.augment and random.random() > 0.5:
            image = self.transform(torch.tensor(image).permute(2, 0, 1)).permute(1, 2, 0).numpy()

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
