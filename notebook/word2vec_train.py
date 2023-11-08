import re
import pickle
from collections import Counter
import numpy as np
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import MeCab
from tqdm.notebook import tqdm
def tokenize_with_mecab(sentences):
    # Initialize MeCab with the specified dictionary
    corpus = []
    for sentence in sentences:
        sentence = re.sub("http://news.livedoor.com/article/detail/[0-9]{7}/","", sentence) # 注2）
        sentence = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{4}","", sentence) # 注3）
        sentence = re.sub("[「」]","", sentence)
        # Parse the sentence
        node = mecab.parseToNode(sentence)
        # Iterate over all nodes
        while node:
            # Extract the surface form of the word
            word = node.surface
            # Skip empty words and add to the corpus
            if word:
                corpus.append(word)
            node = node.next
    return corpus


# コンテキストとターゲットの作成関数の実装
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


with open("/home/lyuzeyu/lyuzeyu/css_nlp/notebook/Data/dokujo-tsushin.txt", mode="r",encoding="utf-8") as f:
    corpus = []
    for line in f:
        cleaned_line = line.replace('\u3000', '').replace('\n', '')
        if cleaned_line!="":
            corpus.append(cleaned_line)
            

from torch.utils.data import Dataset, DataLoader

class CBOWDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]

class SimpleCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SimpleCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        # Embed the input words. 
        # Inputs should have the shape [batch_size, context_size]
        embeds = self.embeddings(inputs)  # Resulting shape [batch_size, context_size, embedding_size]
        
        # Sum the embeddings for each context word to get a single embedding vector per batch sample.
        # The resulting shape should be [batch_size, embedding_size]
        out = torch.sum(embeds, dim=1)
        
        # Pass the summed embeddings through the linear layer
        # The output shape will be [batch_size, vocab_size]
        out = self.linear1(out)
        
        # Apply log softmax to get log probabilities over the vocabulary for each sample in the batch
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Initialize the MeCab tokenizer
#mecab = MeCab.Tagger()
mecab = MeCab.Tagger()
corpus = tokenize_with_mecab(corpus)

word_to_id = {}
id_to_word = {}

for word in corpus:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

# リストに変換
corpus = [word_to_id[word] for word in corpus]

# NumPy配列に変換
corpus = np.array(corpus)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# コンテキストとターゲットを作成
contexts, targets = create_contexts_target(corpus, window_size=2)
contexts = torch.tensor(contexts, dtype=torch.long).to(device)
targets = torch.tensor(targets, dtype=torch.long).to(device)

# Convert contexts and targets to tensors
contexts_tensor = torch.tensor(contexts, dtype=torch.long).to(device)
targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)

# Create the dataset
dataset = CBOWDataset(contexts_tensor, targets_tensor)

# Create the DataLoader
batch_size = 256  # You can adjust the batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# パラメータの設定
embedding_size = 10
learning_rate = 0.01
epochs = 500
vocab_size = len(word_to_id)

# モデルのインスタンス化
model = SimpleCBOW(vocab_size, embedding_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

writer = SummaryWriter('runs/cbow_experiment_2')

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
        writer.add_scalar('Training loss', loss.item(), epoch * len(data_loader) + i)
    # Log the total loss for the epoch
    writer.add_scalar('Total Training loss', total_loss, epoch)
    print(f'Epoch {epoch}, Total loss: {total_loss}')
    
word_embeddings = model.embeddings.weight.data
words = [id_to_word[i] for i in range(len(id_to_word))]

from torch.utils.tensorboard import SummaryWriter

# Initialize the writer
writer = SummaryWriter('runs/cbow_embeddings')

# Add embedding to the writer
writer.add_embedding(word_embeddings, metadata=words)

# Close the writer
writer.close()