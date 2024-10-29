import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

# データセットクラスの定義
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)  # NumPy配列に変換
        self.y = torch.tensor(y.values, dtype=torch.float32)  # NumPy配列に変換

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ネットワークの定義
class TitanicModel(nn.Module):
    def __init__(self, input_size):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 訓練関数
def train(model, dataloader, loss_func, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for X, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# モデル評価関数
def evaluate_model(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch).squeeze()
            predicted_classes = outputs.round().numpy()  # 0または1に丸める
            predictions.extend(predicted_classes)
            true_labels.extend(y_batch.numpy())

    print(classification_report(true_labels, predictions, target_names=["Did not survive", "Survived"]))

# メインの実行部分
if __name__ == "__main__":
    # ここでX_trainとy_trainを定義または読み込む
    # 例えば、pandasを用いてCSVから読み込むなど

    # ダミーデータの生成
    import pandas as pd
    import numpy as np

    # 仮のデータ
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(100, 5))  # 100サンプル、5特徴量
    y_train = pd.Series(np.random.randint(0, 2, size=100))  # 0または1のラベル

    # データセットとデータローダーの作成
    train_dataset = TitanicDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # モデル、損失関数、オプティマイザの定義
    model = TitanicModel(input_size=X_train.shape[1])
    loss_func = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamWオプティマイザ

    # 訓練の実行
    train(model, train_dataloader, loss_func, optimizer, epochs=500)

    # モデル評価
    evaluate_model(model, train_dataloader)