#!/usr/bin/env python
# coding: utf-8

# # Seq2seqの応用：機械翻訳

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import spacy
import numpy as np
import random
import math
import time
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"
path = './Data/raw'


# In[2]:


with open(path, 'r') as f:
  raw_data = f.readlines()
raw_list = [re.sub('\n', '', s).split('\t') for s in raw_data]


# In[3]:


raw_df = pd.DataFrame(raw_list,
                  columns=['English', 'Japanese'])
raw_df.head()


# In[4]:


get_ipython().system('python3 -m spacy download ja_core_news_md')
get_ipython().system('python3 -m spacy download en_core_web_md')


# In[5]:


JA = spacy.load("ja_core_news_md")
EN = spacy.load("en_core_web_md")


# In[6]:


[token.text for token in EN.tokenizer(raw_df["English"][0])]


# In[7]:


[token.text for token in JA.tokenizer(raw_df["Japanese"][0])]


# In[8]:


def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]


# In[18]:


def preprocess_text(text,tokenizer,sos_token="<sos>",eos_token="<eos>"):
    text = text.lower()
    tokens = [tok.text for tok in tokenizer.tokenizer(text)]
    tokens = [sos_token] + tokens + [eos_token]
    return tokens


# In[24]:


raw_df["en_tokens"]=raw_df["English"].apply(lambda x: preprocess_text(x,EN))
raw_df["ja_tokens"]=raw_df["Japanese"].apply(lambda x: preprocess_text(x,JA))


# In[25]:


train_val_df, test_df = train_test_split(raw_df, test_size=0.2)
# Split the training plus validation set into separate training and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=0.25)


# In[26]:


train_df


# In[30]:


min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_df["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

ja_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_df["ja_tokens"],
    min_freq=min_freq,
    specials=special_tokens,  
)


# In[31]:


assert en_vocab[unk_token] == ja_vocab[unk_token]
assert en_vocab[pad_token] == ja_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]


# In[35]:


en_vocab.set_default_index(unk_index)
ja_vocab.set_default_index(unk_index)


# In[38]:


train_df['en_ids'] = train_df["en_tokens"].apply(lambda x: [en_vocab[token] for token in x])
train_df['ja_ids'] = train_df["ja_tokens"].apply(lambda x: [ja_vocab[token] for token in x])


# In[39]:


val_df['en_ids'] = val_df["en_tokens"].apply(lambda x: [en_vocab[token] for token in x])
val_df['ja_ids'] = val_df["ja_tokens"].apply(lambda x: [ja_vocab[token] for token in x])
test_df['en_ids'] = test_df["en_tokens"].apply(lambda x: [en_vocab[token] for token in x])
test_df['ja_ids'] = test_df["ja_tokens"].apply(lambda x: [ja_vocab[token] for token in x])


# In[45]:


val_df.head()


# In[54]:



def create_dataset(df, pad_index):
    # データをテンソルに変換
    en_ids = [torch.LongTensor(ids) for ids in df['en_ids'].tolist()]
    ja_ids = [torch.LongTensor(ids) for ids in df['ja_ids'].tolist()]

    # pad_indexでパディング
    en_ids = pad_sequence(en_ids, batch_first=True, padding_value=pad_index)
    ja_ids = pad_sequence(ja_ids, batch_first=True, padding_value=pad_index)

    # TensorDatasetを作成
    dataset = TensorDataset(en_ids, ja_ids)

    return dataset


# In[62]:


train_data = create_dataset(train_df, pad_index)
val_data = create_dataset(val_df, pad_index)
test_data = create_dataset(test_df, pad_index)


# In[58]:


# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# In[60]:


train_loader.dataset[35]


# In[63]:


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, hidden = self.rnn(embedded) # no cell state in GRU!
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden


# In[64]:


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # context = [n layers * n directions, batch size, hidden dim]
        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hidden dim]
        # context = [1, batch size, hidden dim]
        input = input.unsqueeze(0)
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, embedding dim]
        emb_con = torch.cat((embedded, context), dim = 2)
        #emb_con = [1, batch size, embedding dim + hidden dim]
        output, hidden = self.rnn(emb_con, hidden)
        # output = [seq len, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]
        output = torch.cat((
            embedded.squeeze(0), 
            hidden.squeeze(0), 
            context.squeeze(0)
        ),
            dim=1)
        # output = [batch size, embedding dim + hidden dim * 2]
        prediction = self.fc_out(output)
        # prediction = [batch size, output dim]
        return prediction, hidden


# In[65]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_dim == decoder.hidden_dim,             "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is the context
        context = self.encoder(src)
        # context = [n layers * n directions, batch size, hidden dim]
        # context also used as the initial hidden state of the decoder
        hidden = context
        # hidden = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            # output = [batch size, output dim]
            # hidden = [1, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs


# In[66]:


input_dim = len(ja_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    decoder_dropout,
)


model = Seq2Seq(encoder, decoder, device).to(device)


# In[67]:


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
        
model.apply(init_weights)


# In[68]:


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)


# In[69]:


def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


# In[70]:


def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0) #turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


# In[ ]:


n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
        
    train_loss = train_fn(
        model, 
        train_loader, 
        optimizer, 
        criterion, 
        clip, 
        teacher_forcing_ratio, 
        device,
    )
    
    valid_loss = evaluate_fn(
        model, 
        valid_data_loader, 
        criterion, 
        device,
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut2-model.pt")
    
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

