import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class MyDataset(Dataset):
    def __init__(self, features, target= None):
        self.features = torch.tensor(features, dtype = torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.target is not None:
            return self.features[idx], self.target[idx]
        else:
            return self.features[idx]

def train_val_split(csv_file, test_size= 0.2, random_state=42):
    data = pd.read_csv(csv_file).drop(columns='ID')
    X = data.drop(columns='y').values
    y = data['y'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state = random_state)
    
    # normalization 
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    
    return train_dataset, val_dataset, scaler

def get_data_loader(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader  
