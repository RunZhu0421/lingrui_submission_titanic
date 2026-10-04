import torch
from torch.utils.data import Dataset, DataLoader

class TitanicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def get_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_loader = DataLoader(
        TitanicDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True   # 训练集每次打乱
    )
    test_loader = DataLoader(
        TitanicDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False  # 测试集不用打乱
    )
    return train_loader, test_loader