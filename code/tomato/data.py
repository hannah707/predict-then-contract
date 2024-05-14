import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TomatoPriceDataset(Dataset):
    def __init__(self, features, price_1, price_2, true_yield):
        self.features = torch.from_numpy(features).to(torch.float32)
        self.price_1 = torch.from_numpy(price_1).unsqueeze(1).to(torch.float32)
        self.price_2 = torch.from_numpy(price_2).unsqueeze(1).to(torch.float32)
        self.true_yield = torch.from_numpy(true_yield).unsqueeze(1).to(torch.float32)
        self.indices = list(range(len(features)))
    
    def __getitem__(self, index):
        x = self.features[index]
        y1 = self.price_1[index]
        y2 = self.price_2[index]
        y3 = self.true_yield[index]
        return x.to(device), y1.to(device), y2.to(device), y3.to(device)
    
    def __len__(self):
        return len(self.features)
