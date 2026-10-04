import torch.nn as nn

class TitanicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),   # 全连接层
                nn.BatchNorm1d(h),    # 批归一化（加速训练）
                nn.ReLU(),            # 激活函数（引入非线性）
                nn.Dropout(dropout),  # 随机丢弃 30%（防止死记硬背）
            ]
            prev = h
        layers.append(nn.Linear(prev, 2))  # 最后输出 2 维（遇难/幸存）
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)