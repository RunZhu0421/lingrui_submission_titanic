import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from data_preprocess import preprocess
from dataset import get_loaders
from model import TitanicMLP

# ------- 超参数配置 -------
SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """固定所有随机种子，保证每次跑结果一样"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(SEED)
    os.makedirs("saved", exist_ok=True)
    print(f"使用设备: {DEVICE}")

    # 1. 加载和预处理数据
    df = pd.read_csv("train.csv")
    X, y, feature_cols, stats = preprocess(df, is_train=True)

    # 2. 划分 + 标准化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # 3. 打包数据
    train_loader, test_loader = get_loaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # 4. 定义模型、损失、优化器
    model = TitanicMLP(len(feature_cols)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # 5. 开始训练（循环 200 轮）
    for epoch in range(1, EPOCHS + 1):
        # ---- 训练 ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # ====== 经典的五步曲 ======
            optimizer.zero_grad()          # 1) 清空上一轮残留的梯度
            out = model(xb)                # 2) 前向传播（算预测值）
            loss = criterion(out, yb)      # 3) 算损失（预测 vs 真实）
            loss.backward()                # 4) 反向传播（算梯度）
            optimizer.step()               # 5) 更新参数（把模型变聪明）
            # ==========================
            
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        train_losses.append(total_loss / total)
        train_accs.append(correct / total)

        # ---- 测试 ----
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():  # 测试不需要算梯度，省内存
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += xb.size(0)
        test_losses.append(total_loss / total)
        test_accs.append(correct / total)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | 训练Loss {train_losses[-1]:.4f} 训练Acc {train_accs[-1]:.4f} | "
                  f"测试Loss {test_losses[-1]:.4f} 测试Acc {test_accs[-1]:.4f}")

    # 6. 保存模型和预处理信息
    torch.save(model.state_dict(), "saved/model.pth")
    with open("saved/preprocess.pkl", "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "stats": stats, "scaler": scaler}, f)
    print(f"\n模型已保存到 saved/ 目录")

    # 7. 画图（保存到 docs 目录）
    os.makedirs("../docs", exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend(); plt.grid(True)
    plt.savefig("../docs/loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve")
    plt.legend(); plt.grid(True)
    plt.savefig("../docs/accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"最终测试集准确率: {test_accs[-1]:.4f}")

if __name__ == "__main__":
    train()