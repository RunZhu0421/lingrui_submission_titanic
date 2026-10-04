import pandas as pd
import torch
from data_preprogress_bymyself import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

##设置种子&比例
SEED = 19560929
TEST_SIZE = 0.2

##通过已写函数进行数据处理
df = pd.read_csv("train.csv")
X,y,feature_cols,stats = preprocess(df,is_train=True)

##进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


##对数据进行标准化处理
scaler = StandardScaler()
x_train_sced = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
x_test_sced = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

##对已有的表格进行封装，处理成张量
class TitanicDataset(Dataset):
    ##设定x&y的格式/类型
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    ##计算数据集的长度
    def __len__(self):
        return len(self.X)

    ##传入数据集的各行（训练集输出y，测试集不输出y）
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

##将数据传入核心的神经网络
def get_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    ##传入训练集，打乱顺序提高训练效果
    train_loader = DataLoader(
        TitanicDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    ##传入测试集，不打乱，方便对应答案
    test_loader = DataLoader(
        TitanicDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
