import numpy as np
import torch
import

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW
        img = torch.from_numpy(img).float() / 255.0  # Convert to tensor and normalize to [0, 1]
        if self.transform:
            img = self.transform(img)
        return img, target
