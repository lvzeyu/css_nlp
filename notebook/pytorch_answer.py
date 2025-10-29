import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

# データセットクラス
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ネットワーク定義（隠れ層＋ドロップアウト追加）
class TitanicModel(nn.Module):
    def __init__(self, input_size):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# 訓練関数
def train(model, dataloader, loss_func, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


# モデル評価関数
def evaluate_model(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch).squeeze()
            preds = (outputs > 0.5).numpy()
            predictions.extend(preds)
            true_labels.extend(y_batch.numpy())

    print(classification_report(true_labels, predictions, target_names=["Did not survive", "Survived"]))


# メイン実行部
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # ダミーデータ作成
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(100, 5))  # 100サンプル×5特徴
    y_train = pd.Series(np.random.randint(0, 2, size=100))

    # DataLoader（バッチサイズ64）
    train_dataset = TitanicDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # モデル・損失関数・オプティマイザ定義
    model = TitanicModel(input_size=X_train.shape[1])
    loss_func = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 訓練実行（epoch増加）
    train(model, train_dataloader, loss_func, optimizer, epochs=1000)

    # モデル評価
    evaluate_model(model, train_dataloader)
