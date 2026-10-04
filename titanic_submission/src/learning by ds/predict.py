import pickle
import pandas as pd
import torch

from data_preprocess import preprocess
from model import TitanicMLP

DEVICE = torch.device("cpu")
SAVE_DIR = "saved"

def load_artifacts():
    # 1. 加载之前保存的预处理信息（包含特征列、中位数、标准化器）
    with open(f"{SAVE_DIR}/preprocess.pkl", "rb") as f:
        info = pickle.load(f)
    feature_cols = info["feature_cols"]
    stats = info["stats"]
    scaler = info["scaler"]

    # 2. 加载训练好的模型权重
    model = TitanicMLP(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/model.pth", map_location=DEVICE))
    model.eval()  # 切到预测模式，关闭 Dropout 和 BatchNorm 更新
    return model, feature_cols, stats, scaler

def predict_one(passenger: dict):
    model, feature_cols, stats, scaler = load_artifacts()

    # 1. 把新乘客包装成 DataFrame，并复用训练时的预处理逻辑
    df = pd.DataFrame([passenger])
    X, _, _ = preprocess(df, is_train=False, stats=stats)  # 注意 is_train=False

    # 2. 补齐缺失的特征列（如果新乘客的 Title 是训练集没见过的，补 0）
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # 3. 复用训练时的标准化器进行转换
    X_scaled = scaler.transform(X)
    xb = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    # 4. 喂给模型预测
    with torch.no_grad():
        out = model(xb)
        prob = torch.softmax(out, dim=1)[0]
        pred = int(out.argmax(1).item())

    return pred, float(prob[pred])

if __name__ == "__main__":
    # 测试两个极端样本
    samples = [
        {   # 3 等舱、男性、22 岁、票价便宜
            "Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male",
            "Age": 22, "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171",
            "Fare": 7.25, "Cabin": None, "Embarked": "S",
        },
        {   # 1 等舱、女性、40 岁、票价昂贵
            "Pclass": 1, "Name": "Brown, Mrs. James Joseph", "Sex": "female",
            "Age": 40, "SibSp": 1, "Parch": 0, "Ticket": "PC 17610",
            "Fare": 80.0, "Cabin": "B4", "Embarked": "C",
        },
    ]

    print("=== 新样本预测 ===")
    for s in samples:
        pred, prob = predict_one(s)
        label = "幸存" if pred == 1 else "遇难"
        print(f"{s['Name']:35s} → 预测: {label} ({pred}), 置信度 {prob:.3f}")