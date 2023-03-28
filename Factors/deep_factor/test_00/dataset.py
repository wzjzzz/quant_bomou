from torch.utils.data import Dataset

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