import pandas as pd
from data_preprocess import preprocess

# 读取数据并处理
df = pd.read_csv("train.csv")
X, y, feature_cols, stats = preprocess(df, is_train=True)

print("=== 处理后的特征矩阵形状 ===")
print(X.shape)

print("\n=== 所有特征列名字 ===")
print(feature_cols)

print("\n=== 还有没有缺失值？（必须为 0） ===")
print(X.isnull().sum().sum())

print("\n=== 前 3 行特征数据（全是数字了！） ===")
print(X.head(3))