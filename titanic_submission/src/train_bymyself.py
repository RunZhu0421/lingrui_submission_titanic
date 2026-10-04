import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

## 导入前边的几个程序里的函数和结果
from core_model_bymyself import TitanicMLP

from data_prepare_and_set_bymyself import (
    x_train_sced, x_test_sced, y_train, y_test,
    feature_cols, stats, scaler, get_loaders
)

##定义常量作为超参数
BATCH_SIZE = 64 ##批次大小
EPOCHS = 200 ##训练轮数
LR = 0.001 ##学习率
DEVICE = torch.device("cpu")##注：这里在model_test.py里已经测试过了，我电脑不支持，所以这里直接用cpu


def train():
    ##新建文件夹用来存储信息
    os.makedirs("saved", exist_ok=True)
    os.makedirs("../docs", exist_ok=True)
    ##打印数据
    print(f"使用设备: {DEVICE}")
    print(f"训练集: {x_train_sced.shape}, 测试集: {x_test_sced.shape}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集幸存率: {y_train.mean():.3f}, 测试集幸存率: {y_test.mean():.3f}")

    ##调用data_prepare_and_set_bymyself.py里的get_loaders()函数，获取训练集和测试集的DataLoader
    train_loader, test_loader = get_loaders(
        x_train_sced, y_train, x_test_sced, y_test, BATCH_SIZE
    )

    ##调用model
    model = TitanicMLP(input_dim=len(feature_cols)).to(DEVICE)
    ###掉函数自动算损失函数
    criterion = nn.CrossEntropyLoss()
    ###调函数定义优化器
    optimizer = Adam(model.parameters(), lr=LR)

    ##定义数组记录损失和准确率
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    ##训练的循环
    for epoch in range(1, EPOCHS + 1):
        ##调函数训练
        model.train()
        ##定义初始值
        total_loss, correct, total = 0.0, 0, 0
        ##导入数据
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            ##这个核心步骤貌似是固定的，直接抄的书
            optimizer.zero_grad()          # 1. 清空梯度
            out = model(xb)                # 2. 前向传播
            loss = criterion(out, yb)      # 3. 计算损失
            loss.backward()                # 4. 反向传播
            optimizer.step()               # 5. 更新参数

            ##简单的小求和，计算总值
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        
        ##通过总值平均算出结果并加入数组
        train_losses.append(total_loss / total)
        train_accs.append(correct / total)

        ##进入测试阶段
        model.eval()
        total_loss, correct, total = 0.0, 0, 0 ##重新定义，防止乱套
        ##这里是对电脑的解禁，直接抄的书
        with torch.no_grad():
            ##导入数据*2
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)##算结果
                loss = criterion(out, yb)
                ##依旧求和
                total_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += xb.size(0)
        ##依旧均分
        test_losses.append(total_loss / total)
        test_accs.append(correct / total)
        ##每20个输出一次，便于观察
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | 训练Loss {train_losses[-1]:.4f} 训练Acc {train_accs[-1]:.4f} | "
                  f"测试Loss {test_losses[-1]:.4f} 测试Acc {test_accs[-1]:.4f}")

    ##保存模型数据到刚才创建的文件夹里
    torch.save(model.state_dict(), "saved/model.pth")
    with open("saved/preprocess.pkl", "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "stats": stats, "scaler": scaler}, f)
    print(f"\n模型已保存到 saved/ 目录")

    ##画图
    plt.figure(figsize=(8, 5))##画布大小
    plt.plot(train_losses, label="Train Loss")##画线
    plt.plot(test_losses, label="Test Loss")##画线*2
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")##定义x，y，标题
    plt.legend(); plt.grid(True)##网格背景
    plt.savefig("../docs/loss_curve.png", dpi=200, bbox_inches="tight")##保存+像素要求+去白边
    plt.close()##关闭

    ##原理同上
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve")
    plt.legend(); plt.grid(True)
    plt.savefig("../docs/accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"最终测试集准确率: {test_accs[-1]:.4f}")

##固定用法，自动调用train()函数
if __name__ == "__main__":
    train()