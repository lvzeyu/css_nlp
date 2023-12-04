import pandas as pd
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import re
import nltk
from nltk.tokenize import word_tokenize
import gensim.downloader
word2vec = gensim.downloader.load('word2vec-google-news-300')
# Download NLTK data (if not already done)
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
def create_dataset(df,label):
    features = torch.tensor(df['embedding'].tolist(),dtype=torch.float32).to(device)
    labels = torch.tensor(df[label].values, dtype=torch.long).to(device)
    return TensorDataset(features, labels)

# Function for preprocessing text
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    return tokens

def tokens_to_embedding(tokens, model, embedding_size=300):
    embeddings = [model[word] for word in tokens if word in model] # 単語リストのリストをループし、model[word]で各単語のベクトルを取得し、リストに格納
    # embeddingsが空の場合は、ゼロベクトルを返す
    if len(embeddings) == 0:
        return np.zeros(embedding_size)
    # embeddingsが空でない場合は、ベクトルの平均を返す。その結果は、ベクトルになる、センテンスの埋め込みとして使用できる
    else:
        return np.mean(embeddings, axis=0)

df=pd.read_csv("./Data/sentiment-emotion-labelled_Dell_tweets.csv")
df["sentiment"] = df["sentiment"].replace({"positive":2,"negative":0,"neutral":1})
df['processed_text'] = df['Text'].apply(preprocess_text)
df["embedding"] = df["processed_text"].apply(lambda x: tokens_to_embedding(x, word2vec))

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

batch_size = 32

train_dataset_sentiment = create_dataset(train_df,label="sentiment")
val_dataset_sentiment = create_dataset(val_df,label="sentiment")
test_dataset_sentiment = create_dataset(test_df,label="sentiment")

train_loader_sentiment  = DataLoader(train_dataset_sentiment, batch_size=batch_size, shuffle=True)
val_loader_sentiment  = DataLoader(val_dataset_sentiment, batch_size=batch_size)
test_loader_sentiment  = DataLoader(test_dataset_sentiment, batch_size=batch_size)

class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
    # Set initial hidden and cell states
    # Forward propagate LSTM
        out, _ = self.lstm(x)
        print(out.shape)

    # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

# モデル、損失関数、最適化アルゴリズムの定義
model = LSTMNN(300, 100, 2, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 訓練ループ
for epoch in range(10):
    for i, (tweets, labels) in enumerate(train_loader_sentiment):
        tweets = tweets.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(tweets)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 各エポックの終わりに損失を表示
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')    
