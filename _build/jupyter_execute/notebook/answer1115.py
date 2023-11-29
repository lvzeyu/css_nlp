#!/usr/bin/env python
# coding: utf-8

# # Answer1115
# 
# ## 復習
# 
# ### ニューラルネットワーク
# ニューラルネットワークは、特定の入力と出力の間の複雑な非線形関係をモデル化する能力を持っています。適切に入力と出力を設計することで、様々なモデル構築に柔軟に適用できます。
# 
# - 日英通訳
#     - 入力:日本語トークン
#     - 出力:英語トークン
# - センチメント分類
#     - 入力:テキストトークン
#     - 出力:センチメントラベル
# - CBOW
#     - 入力：周辺単語のトークン
#     - 出力: ターゲット単語のトークン   
# 
# ### 機械学習・深層学習ためテキストの処理
# 
# 機械学習・深層学習はテキストをそのまま扱うことができないので、テキストの「情報」(単語の出現頻度、単語の順番、単語の意味など)を「特徴量」としてまとめる必要があります。
# 
# 特徴量では、テキストの「情報」がより多く、全面的に反映できれば、モデルのパフォーマンスが良くなると考えられます。
# 
# 単語分散表現では、「単語の意味」を数値的に捉えることができますので、テキストの特徴量の作成に有用な方法となっています。
# 
# ### Word2Vec
# 
# Word2Vecとは、単語分散表現（word embedding）を学習するための手法です。
# 
# この手法は、(CBOWの場合)周囲単語からターゲットを予測する「偽タスク」を通じて単語の意味を捉えるように設計されています。
# 
# ターゲットをうまく予測するために、入力テキストの単語を適切に処理し、単語間の関係や意味をより正確に捉えるよう調整する必要があります。このプロセスは、エンコーディング(encoding)と言います。
# 
# モデルが偽タスクでのパフォーマンスを向上させるにつれて、単語意味のエンコーディングもより精度が高くなります。単語意味のエンコーディングは、単語間の意味的および文法的な関係をより良く捉えた単語分散表現として扱えます。
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## データ準備
# 
# ### CSVファイルを読み込む

# In[2]:


df=pd.read_csv("./Data/sentiment-emotion-labelled_Dell_tweets.csv")
df.head()


# ### データの確認

# - Text: テキストデータ
#     - デキストクリーニング
#     - 学習済みword2vecモデルで単語分散表現に変換する $\to$ 機械学習・深層学習モデルの入力として
# - sentiment
#     - ラベルデータを数値形式に変換する
#     - ラベルデータによって機械学習・深層学習モデルの出力層を調整する　$\to$　出力層の出力とラベルデータを比較し、交差エントロピーを計算する
# - emotion
#     - ラベルデータを数値形式に変換する
#     - ラベルデータによって機械学習・深層学習モデルの出力層を調整する　$\to$　出力層の出力とラベルデータを比較し、交差エントロピーを計算する

# ### ラベルデータの処理

# #### sentimentラベル

# In[3]:


df["sentiment"].value_counts()


# In[4]:


df["sentiment"] = df["sentiment"].replace({"positive":2,"negative":0,"neutral":1})


# In[5]:


df["sentiment"].value_counts()


# #### emotionラベル

# In[6]:


df["emotion"].value_counts()


# In[7]:


unique_emotions = df['emotion'].unique()
emotion_dict = {emotion: index for index, emotion in enumerate(unique_emotions)}
emotion_dict


# In[8]:


df["emotion"]=df["emotion"].replace(emotion_dict)


# In[9]:


df["emotion"].value_counts()


# ### テキストデータの前処理
# 
# - テキストを小文字に変換
# - 句読点を削除
# - トークン化

# In[10]:


import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data (if not already done)
nltk.download('punkt')

# Function for preprocessing text
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    return tokens


# In[11]:


df['processed_text'] = df['Text'].apply(preprocess_text)


# In[12]:


df.head()


# ### 単語分散表現によって特徴量の作成

# In[13]:


import gensim.downloader
word2vec = gensim.downloader.load('word2vec-google-news-300')


# In[17]:


def tokens_to_embedding(tokens, model, embedding_size=300):
    embeddings = [model[word] for word in tokens if word in model] # 単語リストのリストをループし、model[word]で各単語のベクトルを取得し、リストに格納
    # embeddingsが空の場合は、ゼロベクトルを返す
    if len(embeddings) == 0:
        return np.zeros(embedding_size)
    # embeddingsが空でない場合は、ベクトルの平均を返す。その結果は、1次元のベクトルになる、センテンスの埋め込みとして使用できる
    else:
        return np.mean(embeddings, axis=0)


# In[18]:


df["embedding"] = df["processed_text"].apply(lambda x: tokens_to_embedding(x, word2vec))


# In[19]:


df.head()


# ### トレーニング、バリデーション、テストデータに分割

# In[20]:


from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


# ### 学習用データセットの作成(Batch Datasets)

# In[24]:


from torch.utils.data import DataLoader, TensorDataset

def create_dataset(df,label):
    features = torch.tensor(df['embedding'].tolist(),dtype=torch.float32).to(device)
    labels = torch.tensor(df[label].values, dtype=torch.long).to(device)
    return TensorDataset(features, labels)


# #### sentimentデータセット

# In[27]:


batch_size = 32

train_dataset_sentiment = create_dataset(train_df,label="sentiment")
val_dataset_sentiment = create_dataset(val_df,label="sentiment")
test_dataset_sentiment = create_dataset(test_df,label="sentiment")


# In[28]:


train_loader_sentiment  = DataLoader(train_dataset_sentiment, batch_size=batch_size, shuffle=True)
val_loader_sentiment  = DataLoader(val_dataset_sentiment, batch_size=batch_size)
test_loader_sentiment  = DataLoader(test_dataset_sentiment, batch_size=batch_size)


# #### emotionデータセット

# In[29]:


batch_size = 32

train_dataset_emotion = create_dataset(train_df,label="emotion")
val_dataset_emotion = create_dataset(val_df,label="emotion")
test_dataset_emotion = create_dataset(test_df,label="emotion")


# In[30]:


train_loader_emotion  = DataLoader(train_dataset_emotion, batch_size=batch_size, shuffle=True)
val_loader_emotion  = DataLoader(val_dataset_emotion, batch_size=batch_size)
test_loader_emotion  = DataLoader(test_dataset_emotion, batch_size=batch_size)


# ## 学習の実行

# ### ニューラルネットワークモデルの作成

# In[31]:


import torch.nn as nn
import torch.optim as optim

# Define a simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 入力と出力の次元数はデータに合わせて設定する
# 
# - 入力の次元数は単語分散表現の次元数(今回は$300$)は一致しています
#     - 一つの単語トークンの単語分散表現は$300$のため、複数単語分散表現の平均(テキストの特徴量)の次元数も$300$です
# - 出力の次元数はラベルデータのカテゴリ数と一致しています $\to$ 交差エントロピーを計算するため

# In[37]:


df["embedding"][0].shape


# ### sentimentモデル
# 
# #### sentimentモデルの学習

# In[42]:


label_size = df["sentiment"].nunique()
embedding_size = 300
model = SimpleNN(input_size=embedding_size, hidden_size=100, num_classes=label_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[43]:


from sklearn.metrics import accuracy_score, f1_score

num_epochs = 50
best_f1_score = 0.0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader_sentiment:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader_sentiment:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.numpy())
            val_labels.extend(labels.numpy())
    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
 


# #### sentimentモデルの検証

# In[44]:


model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader_sentiment:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.numpy())
        test_labels.extend(labels.numpy())


# In[45]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap="crest")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# ### emotionモデル
# 
# #### emotionモデルの学習

# In[47]:


label_size = df["emotion"].nunique()
embedding_size = 300
model = SimpleNN(input_size=embedding_size, hidden_size=100, num_classes=label_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[48]:


num_epochs = 50
best_f1_score = 0.0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader_emotion:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader_emotion:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.numpy())
            val_labels.extend(labels.numpy())
    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')


# #### emotionモデルの検証

# In[49]:


model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader_emotion:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.numpy())
        test_labels.extend(labels.numpy())


# In[51]:


cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap="crest")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# 
