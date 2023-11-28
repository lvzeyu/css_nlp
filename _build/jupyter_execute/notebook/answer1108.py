#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# In[2]:


# 前処理関数の実装
def preprocess(text):
    # 前処理
    text = text.lower() # 小文字に変換
    text = text.replace('.', ' .') # ピリオドの前にスペースを挿入
    words = text.split(' ') # 単語ごとに分割
    
    # ディクショナリを初期化
    word_to_id = {}
    id_to_word = {}
    
    # 未収録の単語をディクショナリに格納
    for word in words:
        if word not in word_to_id: # 未収録の単語のとき
            # 次の単語のidを取得
            new_id = len(word_to_id)
            
            # 単語をキーとして単語IDを格納
            word_to_id[word] = new_id
            
            # 単語IDをキーとして単語を格納
            id_to_word[new_id] = word
    
    # 単語IDリストを作成
    corpus = [word_to_id[w] for w in words]
    
    return corpus, word_to_id, id_to_word


# In[3]:


def create_contexts_target(corpus, window_size=1):
    
    # ターゲットを抽出
    target = corpus[window_size:-window_size]
    
    # コンテキストを初期化
    contexts = []
    
    # ターゲットごとにコンテキストを格納
    for idx in range(window_size, len(corpus) - window_size):
        
        # 現在のターゲットのコンテキストを初期化
        cs = []
        
        # 現在のターゲットのコンテキストを1単語ずつ格納
        for t in range(-window_size, window_size + 1):
            
            # 0番目の要素はターゲットそのものなので処理を省略
            if t == 0:
                continue
            
            # コンテキストを格納
            cs.append(corpus[idx + t])
            
        # 現在のターゲットのコンテキストのセットを格納
        contexts.append(cs)
    
    # NumPy配列に変換
    return np.array(contexts), np.array(target) 


# In[4]:


text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold."""


# In[5]:


corpus, word_to_id, id_to_word = preprocess(text)
contexts, targets = create_contexts_target(corpus, window_size=2)


# In[6]:


class CBOWDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


# In[7]:


class SimpleCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SimpleCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = torch.sum(embeds, dim=1)
        out = self.linear1(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# In[8]:


device='cuda' if torch.cuda.is_available() else 'cpu'
# Convert contexts and targets to tensors
contexts_tensor = torch.tensor(contexts, dtype=torch.long).to(device)
targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)

# Create the dataset
dataset = CBOWDataset(contexts_tensor, targets_tensor)


# In[9]:


# Create the DataLoader
batch_size = 10  # You can adjust the batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[10]:


# パラメータの設定
embedding_size = 10
learning_rate = 0.01
epochs = 100
vocab_size = len(word_to_id)

# モデルのインスタンス化
model = SimpleCBOW(vocab_size, embedding_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Training loop with batch processing
for epoch in range(epochs):
    total_loss = 0
    for i, (context_batch, target_batch) in enumerate(data_loader):
        # Zero out the gradients from the last step
        model.zero_grad()
        # Forward pass through the model
        log_probs = model(context_batch)
        # Compute the loss
        loss = loss_function(log_probs, target_batch)
        # Backward pass to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
        # Accumulate the loss
        total_loss += loss.item()
    # Log the total loss for the epoch
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Total loss: {total_loss}')


# In[ ]:




