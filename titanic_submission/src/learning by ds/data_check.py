import pandas as pd

df = pd.read_csv("train.csv")

print("=== 数据形状（行, 列） ===")
print(df.shape)

print("\n=== 前5行数据 ===")
print(df.head())

print("\n=== 各列缺失值数量 ===")
print(df.isnull().sum())

print("\n=== 幸存者分布（0=遇难, 1=幸存） ===")
print(df["Survived"].value_counts())