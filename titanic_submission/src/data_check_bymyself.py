import pandas as pd
df = pd.read_csv("train.csv")

print("\n前五行")
print(df.head())

print("\n数据形状")
print(df.shape)

print("\n所有列名")
print(df.columns)

print("\n所有缺失值")
print(df.isnull().sum())

print("\n生存情况")
print(df["Survived"].value_counts())