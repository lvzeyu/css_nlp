#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import re
import pickle
from janome.tokenizer import Tokenizer
import numpy as np
import collections

with open("/Users/ryozawau/css_nlp/notebook/Data/dokujo-tsushin.txt", mode="r",encoding="utf-8") as f: # 注1）
    original_corpus = f.readlines()

text = re.sub("http://news.livedoor.com/article/detail/[0-9]{7}/","", original_corpus) # 注2）
text = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{4}","", text) # 注3）
text = re.sub("[\f\n\r\t\v]","", text)
text = re.sub("　","", text)
text = re.sub("[「」]","", text)
text = [re.sub("[（）]","", text)]

# ＜ポイント＞
t = Tokenizer()

words_list = []
for word in text:
    words_list.append(t.tokenize(word, wakati=True))


# In[65]:


with open("/Users/ryozawau/css_nlp/notebook/Data/dokujo-tsushin.txt", mode="r",encoding="utf-8") as f: # 注1）
    original_corpus = f.readlines()


# In[66]:


import MeCab
from tqdm.notebook import tqdm
def tokenize_with_mecab(sentences):
    # Initialize MeCab with the specified dictionary
    corpus = []
    for sentence in sentences:
        sentence = re.sub("http://news.livedoor.com/article/detail/[0-9]{7}/","", sentence) # 注2）
        sentence = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{4}","", sentence) # 注3）
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


# Initialize the MeCab tokenizer
#mecab = MeCab.Tagger()
path = "-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd"
mecab = MeCab.Tagger(path)
corpus = tokenize_with_mecab(original_corpus)


# In[67]:


words


# In[68]:


word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
        
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[1]:', id_to_word[1])
print('id_to_word[2]:', id_to_word[2])
print()
print("word_to_id['女']:", word_to_id['女'])
print("word_to_id['結婚']:", word_to_id['結婚'])
print("word_to_id['夫']:", word_to_id['夫'])


# In[69]:


# 共起行列の作成
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

# ベクトル間の類似度（cos類似度）判定
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

# ベクトル間の類似度をランキング
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

# 正の相互情報量（PPMI）を使用した単語の関連性指標の改善
def ppmi(C, verbose=False, eps = 1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M


# In[70]:


window_size = 2
wordvec_size = 100
vocab_size = len(word_to_id)


# In[71]:


# リストに変換
corpus = [word_to_id[word] for word in words]

# NumPy配列に変換
corpus = np.array(corpus)


# In[72]:


vocab_size


# In[73]:


print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)


# In[49]:


print('calculating PPMI ...')
W = ppmi(C, verbose=True)


# In[74]:


W = np.load("/Users/ryozawau/css_nlp/notebook/Data/W.npy")


# In[75]:


from sklearn.utils.extmath import randomized_svd
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)


# In[51]:


word_vecs = U[:, :wordvec_size]


# In[63]:


querys = ['女性', '結婚', '彼', "秋"]

for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


# In[ ]:


import pickle


# In[58]:


W


# In[59]:


np.save('./Data/W.npy', W)


# In[ ]:


torch.softmax(s, dim=1)
t = torch.tensor([[0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
loss = nn.CrossEntropyLoss()
loss(s,t)


# In[ ]:





# In[ ]:





# In[2]:


import re
import nltk
nltk.download('brown')
from nltk.corpus import brown
import itertools
corpus = []

for cat in ['news']:
    for text_id in brown.fileids(cat):
        raw_text = list(itertools.chain.from_iterable(brown.sents(text_id)))
        text = ' '.join(raw_text)
        text = text.lower()
        text.replace('\n', ' ')
        text = re.sub('[^a-z ]+', '', text)
        corpus.append([w for w in text.split() if w != ''])


# In[4]:


len(corpus)


# In[6]:


len(corpus[0])


# In[ ]:


test_sentence = """When forty winters shall besiege thy brow,
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
And see thy blood warm when thou feel'st it cold.""".split()


# In[ ]:


word_to_id = {}
id_to_word = {}
    
# 未収録の単語をディクショナリに格納
for word in test_sentence:
    if word not in word_to_id: # 未収録の単語のとき
            # 次の単語のidを取得
        new_id = len(word_to_id)
            
            # 単語をキーとして単語IDを格納
        word_to_id[word] = new_id
            
            # 単語IDをキーとして単語を格納
        id_to_word[new_id] = word
    
    # 単語IDリストを作成
corpus = [word_to_id[w] for w in test_sentence]


# In[ ]:


contexts, targets = create_contexts_target(corpus, window_size=2)
print(contexts)
print(targets)


# In[ ]:


# Convert contexts and targets to tensors
contexts_tensor = torch.tensor(contexts, dtype=torch.long).to(device)
targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)

# Create the dataset
dataset = CBOWDataset(contexts_tensor, targets_tensor)


# In[ ]:


# Create the DataLoader
batch_size = 12  # You can adjust the batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


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


model.most_similar(positive=['woman', 'king'], negative=['man'])

