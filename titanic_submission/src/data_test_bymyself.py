import pandas as pd
from data_preprogress_bymyself import preprocess
df = pd.read_csv("train.csv")
X,y,feature_cols,stats = preprocess(df,is_train=True)
print(X.shape)

print(X.isnull().sum().sum())