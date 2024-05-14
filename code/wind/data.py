import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WindPriceDataset(Dataset):
    def __init__(self, features, price_1, price_2, true_yield, pred_yield=[]):
        self.features = torch.from_numpy(features).to(torch.float32)
        self.price_1 = torch.from_numpy(price_1).unsqueeze(1).to(torch.float32)
        self.price_2 = torch.from_numpy(price_2).unsqueeze(1).to(torch.float32)
        self.true_yield = torch.from_numpy(true_yield).unsqueeze(1).to(torch.float32)
        if len(pred_yield)==0:
            pred_yield = np.load('../../inputs/wind/forecast.npy')
            self.pred_yield = torch.from_numpy(pred_yield).unsqueeze(1).to(torch.float32)
        else:
            self.pred_yield = torch.from_numpy(pred_yield).unsqueeze(1).to(torch.float32)
        self.indices = list(range(len(features)))
    
    def __getitem__(self, index):
        x = self.features[index]
        y1 = self.price_1[index]
        y2 = self.price_2[index]
        y3 = self.true_yield[index]
        z = self.pred_yield[index]
        return x.to(device), y1.to(device), y2.to(device), y3.to(device), z.to(device)
    
    def __len__(self):
        return len(self.features)
    
class WindDecisionDataset(Dataset):
    def __init__(self, features, price_1, price_2, true_yield, x_opt):
        self.features = torch.from_numpy(features).to(torch.float32)
        self.price_1 = torch.from_numpy(price_1).unsqueeze(1).to(torch.float32)
        self.price_2 = torch.from_numpy(price_2).unsqueeze(1).to(torch.float32)
        self.true_yield = torch.from_numpy(true_yield).unsqueeze(1).to(torch.float32)
        self.x_opt = torch.from_numpy(x_opt).unsqueeze(1).to(torch.float32)
    
    def __getitem__(self, index):
        x = self.features[index]
        y1 = self.price_1[index]
        y2 = self.price_2[index]
        y3 = self.true_yield[index]
        z = self.x_opt[index]
        return x.to(device), y1.to(device), y2.to(device), y3.to(device), z.to(device)
    
    def __len__(self):
        return len(self.features)