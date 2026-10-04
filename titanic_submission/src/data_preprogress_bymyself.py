import pandas as pd
import re
import numpy as np

##包装加工函数
def preprocess(df,is_train=True,stats=None):
    df = df.copy()

    ##创建新特征：家庭总数
    df["Family_size"] = df["SibSp"] + df["Parch"] + 1
    df["Alone"] = (df["Family_size"] == 1).astype(int)

    ##填补空缺船舱值+提取首字母
    df["Cabin"] = df["Cabin"].fillna("U")
    df["Cabin"] = df["Cabin"].str[0]

    ##设置缺失值
    if is_train:
        stats = {
            "age_mid" : df["Age"].median(),
            "fare_mid" : df["Fare"].median(),
            "embarked_mode" : df["Embarked"].mode()[0]
        }
    df["Age"] = df["Age"].fillna(stats["age_mid"])
    df["Fare"] = df["Fare"].fillna(stats["fare_mid"]) 
    df["Embarked"] = df["Embarked"].fillna(stats["embarked_mode"])

    ##部分特征的数字化
    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    df["Embarked"] = df["Embarked"].map({"C":0,"Q":1,"S":2}).astype(int)

    ##通过独热编码实现对仓号的数字化，并实现拼接
    Cabin_num = pd.get_dummies(df["Cabin"],prefix="Cabin")
    df = pd.concat([df,Cabin_num],axis=1)

    feature_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Family_size","Alone"] + list(Cabin_num.columns) 
    
    X = df[feature_cols].astype(np.float32)
    if is_train:
        y = df["Survived"].astype(np.int64)
        return X,y,feature_cols,stats
    else:
        return X,feature_cols,stats