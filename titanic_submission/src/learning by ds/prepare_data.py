import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocess import preprocess

SEED = 42
TEST_SIZE = 0.2

# 1. 加载并预处理数据
df = pd.read_csv("train.csv")
X, y, feature_cols, stats = preprocess(df, is_train=True)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
print(f"训练集幸存率: {y_train.mean():.3f}, 测试集幸存率: {y_test.mean():.3f}")

# 3. 标准化（极其重要！）
scaler = StandardScaler()

# 注意：只能 fit_transform 训练集！
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
# 测试集只能 transform，绝对不能 fit！
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

print("\n=== 标准化后训练集前 2 行 ===")
print(X_train_scaled.head(2))