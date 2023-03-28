from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def append(self, x, y):
        self.x = torch.concat([self.x, x])
        self.y = torch.concat([self.y, y])
