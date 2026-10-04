import torch.nn as nn

class TitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        ##第一层神经网
        self.l1 = nn.Linear(input_dim, 64)
        ##第二层神经网
        self.l2 = nn.Linear(64, 32)
        ###第三层神经网
        ## self.l3 = nn.Linear(32, 16)
        ###输出层
        self.output = nn.Linear(32, 2)

        ##激活函数&丢弃神经元
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        ## x = self.l3(x)
        ## x = self.relu(x)
        x = self.output(x)
        return x