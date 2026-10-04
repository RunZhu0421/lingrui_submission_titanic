import pickle
import pandas as pd
import torch

##导入之前的函数
from core_model_bymyself import TitanicMLP
from data_preprogress_bymyself import preprocess

##设置超参数
DEVICE = torch.device("cpu")
SAVE_DIR = "saved"


##读取训练过程中的数据
def load_artifacts():
    with open(f"{SAVE_DIR}/preprocess.pkl", "rb") as f:
        info = pickle.load(f)

    feature_cols = info["feature_cols"]
    stats = info["stats"]
    scaler = info["scaler"]

    ##读取训练好的权重，导入数据到新的表
    model = TitanicMLP(input_dim=len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/model.pth", map_location=DEVICE))
    model.eval() ##关丢弃
    return model, feature_cols, stats, scaler


##预测单个乘客
def predict_one(passenger: dict):
    model, feature_cols, stats, scaler = load_artifacts()

    ##把一个乘客的信息，包装成dataframe格式的一行
    df = pd.DataFrame([passenger])

    ##调用之前的处理函数对新的数据进行处理
    X, _, _ = preprocess(df, is_train=False, stats=stats)

    ##处理新数据的特殊列，将之前未处理的列设为0
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    ##调用之前的标准化函数
    X_scaled = scaler.transform(X)
    xb = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    ##调用函数运算结果
    with torch.no_grad():
        out = model(xb)
        prob = torch.softmax(out, dim=1)[0]    ##把得分转成概率
        pred = int(out.argmax(1).item())       ##取概率大的那个作为结果

    return pred, float(prob[pred])


##测试（这个测试的数据是ds给的）
if __name__ == "__main__":
    samples = [
        {   # 3 等舱、男性、22 岁、票价便宜 → 历史上大概遇难
            "Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male",
            "Age": 22, "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171",
            "Fare": 7.25, "Cabin": None, "Embarked": "S",
        },
        {   # 1 等舱、女性、40 岁、票价昂贵 → 历史上大概率幸存
            "Pclass": 1, "Name": "Brown, Mrs. James Joseph", "Sex": "female",
            "Age": 40, "SibSp": 1, "Parch": 0, "Ticket": "PC 17610",
            "Fare": 80.0, "Cabin": "B4", "Embarked": "C",
        },
    ]

    print("新样本预测结果：")
    for s in samples:
        pred, prob = predict_one(s)
        label = "幸存" if pred == 1 else "遇难"
        print(f"{s['Name']:35s} → 预测: {label} ({pred}), 置信度 {prob:.3f}")