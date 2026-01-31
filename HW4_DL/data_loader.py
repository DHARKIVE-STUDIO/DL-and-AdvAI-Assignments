from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

# A simple custom transformation function
def square(x):
    return x ** 2

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.data = torch.arange(1, 101, dtype=torch.float32)  # Numbers 1 to 100 as dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)  # Fix: Return dataset size

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)  # Apply transformation if provided
        return sample
  
# A custom transformation that rotates images by an angle
class RandomRotationTransform:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, x):
        return transforms.functional.rotate(x, self.degree)
