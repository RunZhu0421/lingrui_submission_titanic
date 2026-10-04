import re
import numpy as np
import pandas as pd

def extract_title(name):
    """从 'Braund, Mr. Owen Harris' 提取 'Mr'"""
    match = re.search(r" ([A-Za-z]+)\.", name)
    return match.group(1) if match else ""

def preprocess(df, is_train=True, stats=None):
    df = df.copy()  # 不修改原始数据

    # 1. 提取 Title（称呼：Mr、Mrs 等）
    df["Title"] = df["Name"].apply(extract_title)
    title_map = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
        "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
        "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
        "Capt": "Rare", "Sir": "Rare",
    }
    df["Title"] = df["Title"].map(title_map).fillna("Rare")

    # 2. 家庭特征（单打独斗还是全家出动？）
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 3. Cabin 提取甲板首字母
    df["Cabin"] = df["Cabin"].fillna("U")
    df["CabinDeck"] = df["Cabin"].str[0]

    # 4. 缺失值处理（核心：如果是训练集，算出中位数存起来；测试集直接复用）
    if is_train:
        stats = {
            "age_median": df["Age"].median(),
            "fare_median": df["Fare"].median(),
            "embarked_mode": df["Embarked"].mode()[0],
        }
    df["Age"] = df["Age"].fillna(stats["age_median"])
    df["Fare"] = df["Fare"].fillna(stats["fare_median"])
    df["Embarked"] = df["Embarked"].fillna(stats["embarked_mode"])

    # 5. 类别编码（把文字变成数字）
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    # 6. 独热编码（One-Hot：把 Title 和 Deck 变成多个 0/1 列）
    title_dummies = pd.get_dummies(df["Title"], prefix="Title")
    deck_dummies = pd.get_dummies(df["CabinDeck"], prefix="Deck")
    df = pd.concat([df, title_dummies, deck_dummies], axis=1)

    # 7. 挑选最终的入模特征
    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "Embarked",
    ] + list(title_dummies.columns) + list(deck_dummies.columns)

    X = df[feature_cols].astype(np.float32)

    if is_train:
        y = df["Survived"].astype(np.int64)
        return X, y, feature_cols, stats
    else:
        return X, feature_cols, stats