#!/usr/bin/env python
# coding: utf-8

# # LSTMによる文書分類

# In[1]:


import pandas as pd
import numpy as np
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torchtext.vocab import vocab
from sklearn.metrics import accuracy_score, f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## データ準備
# 
# ### CSVファイルを読み込む

# In[2]:


df= pd.read_csv('./Data/twitter_training.csv',names=['index','brand','sentiment','text'])
df.head()


# ### ラベルデータの処理

# In[3]:


df["label"]=df["sentiment"].replace({"Positive":2,"Negative":0,"Neutral":1,"Irrelevant":np.nan})
df.dropna(inplace=True)
df.head()


# ### テキストデータの前処理
# 
# - テキストを小文字に変換
# - 句読点を削除
# - トークン化
# - 単語ID化

# #### Tokenization

# In[4]:


def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    return tokens


# In[5]:


df["processed_text"]=df["text"].apply(preprocess_text)


# In[6]:


df.head()


# #### 単語辞書
# 
# `Vocab`は、各単語（トークン）に対して一意のインデックス（またはID）を割り当てます。このマッピングにより、テキストデータを数値データに変換することができます。

# In[7]:


counter = Counter()
for line in df["processed_text"]:
    counter.update(line)
Vocab = vocab(counter, min_freq=1)


# In[8]:


# 単語からインデックスへのマッピング
word_to_index = Vocab.get_stoi()

# 最初の5つのアイテムを取得して表示
for i, (word, index) in enumerate(word_to_index.items()):
    if i >= 5:  # 最初の5つのアイテムのみ表示
        break
    print(f"'{word}': {index}")


# In[9]:


df['numericalized_text'] = df["processed_text"].apply(lambda x: [Vocab[token] for token in x])


# In[10]:


df.head()


# In[11]:


def pad_sequences(seq, max_len):
    padded = np.zeros((max_len,), dtype=np.int64)
    if len(seq) > max_len: padded[:] = seq[:max_len]
    else: padded[:len(seq)] = seq
    return padded


# #### Padding
# 
# ニューラルネットワークは、入力データが固定長であることを前提としていますので、テキストシーケンスを特定の最大長にパディング（埋める）する必要があります。

# In[12]:


df["text_length"]=df["numericalized_text"].apply(lambda x: len(x)) 


# In[13]:


df["text_length"].describe()


# In[14]:


max_len=30
df['padded_text'] = df['numericalized_text'].apply(lambda x: pad_sequences(x, max_len))


# In[15]:


df.head()


# ## 学習用データセットの作成(Batch Datasets)

# In[16]:


# Split the original dataset into training plus validation and testing sets
train_val_df, test_df = train_test_split(df, test_size=0.2)

# Split the training plus validation set into separate training and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=0.25)


# In[17]:


# Create TensorDatasets
train_data = TensorDataset(torch.LongTensor(train_df['padded_text'].tolist()), torch.LongTensor(train_df['label'].tolist()))
val_data = TensorDataset(torch.LongTensor(val_df['padded_text'].tolist()), torch.LongTensor(val_df['label'].tolist()))
test_data = TensorDataset(torch.LongTensor(test_df['padded_text'].tolist()), torch.LongTensor(test_df['label'].tolist()))

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# ## モデルの作成
# 
# ### メソッドの説明
# 
# #### `nn.Embedding`
# 
# - `nn.Embedding`は単語の埋め込みを行うために使用されます。単語の埋め込みとは、単語を固定長のベクトルに変換することを指します。このベクトルは、単語の意味的な特性を捉えることができます。`nn.Embedding`の主なパラメータは以下の通りです：
#     - `num_embeddings`：埋め込みを行う単語の総数。通常は語彙のサイズに設定します。
#     - `embedding_dim`：各単語の埋め込みベクトルの次元数。
# - `nn.Embedding`は、整数のインデックスを入力として受け取り、それに対応する埋め込みベクトルを出力します。
# - 下の例では、`input`の各インデックスが対応する埋め込みベクトルに置き換えられ、`embedded`はサイズ`(batch_size, sequence_length, embedding_dim)`のテンソルになります。
# 
# #### `nn.Dropout`
# 
# - ドロップアウトは、ニューラルネットワークの訓練中にランダムにノードを「ドロップアウト」（つまり無効化）することで、過学習を防ぐための一般的なテクニックです`nn.Dropout`の主なパラメータは以下の通りです：
#     - `p`：ノードをドロップアウトする確率。0.0（ノードをドロップアウトしない）から1.0（全てのノードをドロップアウトする）までの値を取ります。デフォルトは0.5です。
# - `nn.Dropout`は、訓練中にのみドロップアウトを適用し、評価（つまりモデルが`.eval()`モードにあるとき）中にはドロップアウトを適用しません。これは、訓練中にはモデルのロバスト性を向上させるためにランダム性が必要である一方、評価中にはモデルの全ての学習特性を使用して一貫した出力を得る必要があるためです。

# In[18]:


embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
embedded = embedding(input)
embedded.shape


# ### モデルの定義
# 
# `hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))`
# 
# ここでは、双方向LSTMの最後の隠れ状態を取り扱っています。
# 
# 双方向LSTMは、順方向と逆方向の2つのLSTMを使用します。順方向のLSTMはシーケンスを通常の順序で処理し、逆方向のLSTMはシーケンスを逆順で処理します。その結果、各時間ステップで2つの隠れ状態（順方向と逆方向のそれぞれから1つずつ）が得られます。
# 
# - `hidden[-2,:,:]`と`hidden[-1,:,:]`は、それぞれ最後の時間ステップでの順方向と逆方向の隠れ状態を取得しています。
# 
# - `torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)`は、これら2つの隠れ状態を結合しています。結合は`dim=1`（つまり、特徴量の次元）に沿って行われます。
# 
# その結果、順方向と逆方向の隠れ状態が1つのベクトルに結合され、そのベクトルは次の全結合層に入力されます。
# 
# `self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)`
# 
# `self.fc`は全結合層で、LSTMからの出力を最終的な出力次元に変換します。この出力は、分類タスクのクラス数に等しいなります。
# 
# 全結合層の入力次元は、LSTMの隠れ状態の次元数に依存します。
# 
# - LSTMが双方向の場合（`bidirectional=True`）、順方向と逆方向の隠れ状態が結合されるため、隠れ状態の次元数は`hidden_dim * 2`になります。
# - LSTMが一方向の場合（`bidirectional=False`）、隠れ状態の次元数は`hidden_dim`になります。
# 
# したがって、`nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)`は、LSTMの方向性に応じて全結合層の入力次元を適切に設定します。
# 
# 出力次元`output_dim`は、タスクのクラス数または回帰の出力次元に設定します。
# 
# `batch_first=True`
# 
# `batch_first=True`を設定すると、
# 
# - 入力テンソルの形状は`(batch_size, sequence_length, input_size)`と解釈されます。つまり、バッチの次元が最初に来ます。
# - `output`テンソルの形状は`(batch_size, seq_len, num_directions * hidden_size)`になります。
# 
# `batch_first=True`を使用する主な理由は、多くの場合、バッチの次元を最初に持ってくると、テンソル操作が直感的になり、コードが読みやすくなります。

# In[19]:


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.bidirectional = bidirectional
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1,:,:]
        return self.fc(hidden.squeeze(0))


# In[20]:


vocab_size = len(Vocab)
embedding_dim = 100  
hidden_dim = 256     
output_dim = 3 
n_layers = 2        
bidirectional = True 
dropout = 0.2        

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)


# In[21]:


model = model.to(device)


# In[22]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[23]:


def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_labels = []
            val_preds = []
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts)
                val_labels.extend(labels.tolist())
                val_preds.extend(torch.argmax(predictions, dim=1).tolist())

            accuracy = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='weighted')
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}, F1 Score: {f1}")
        model.train()


# In[24]:


from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, tensorboard=False, tensorboard_path='./runs'):
    # Initialize TensorBoard writer if tensorboard logging is enabled
    writer = SummaryWriter(tensorboard_path) if tensorboard else None

    model.train()
    for epoch in range(n_epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_labels = []
            val_preds = []
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts)
                val_labels.extend(labels.tolist())
                val_preds.extend(torch.argmax(predictions, dim=1).tolist())

            accuracy = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='weighted')

            # Log metrics to TensorBoard
            if tensorboard:
                writer.add_scalar('Loss/train', loss.item(), epoch)
                writer.add_scalar('Accuracy/val', accuracy, epoch)
                writer.add_scalar('F1-Score/val', f1, epoch)

            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}, F1 Score: {f1}")

        model.train()

    # Close the TensorBoard writer
    if tensorboard:
        writer.close()


# In[25]:


# Train the model
n_epochs = 30
#train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, tensorboard=True, tensorboard_path='./runs/lstm')


# In[ ]:




