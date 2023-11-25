#!/usr/bin/env python
# coding: utf-8

# # word2vec

# 前章では、「カウントベースの手法」によって単語分散表現を得ました。具体的には、単語の共起行列を作り、その行列に対してSVDを適用することで、密なベクトくー 単語分散表現ーを獲得したのです。
# 
# しかし、カウントベースの手法にはいくつかの問題点があります。
# 
# - 大規模なコーパスを扱う場合、巨大な共起行列に対してSVDを計算することが難しい。
# - コーパスの全体から一回の学習で単語分散表現を獲得していますので、新しい単語が追加される場合、再度最初から学習を行う必要があり、単語分散表現更新の効率が低い。
# 
# 「カウントベースの手法」に代わる強力な手法として「推論ベース」の手法が挙げられます。特に、Mikolov et al. {cite}`mikolov-etal-2013-linguistic` {cite}`NIPS2013_9aa42b31`　によって提案されたword2vecの有用性が多くの自然言語処理タスクにおいて示されてきたのです。
# 
# 本章では、word2vecの仕組みについて説明し、それを実装することで理解を深めます。

# ## 推論ベース手法とニューラルネットワーク
# 
# 推論ベースの手法は、ミニバッチで学習する形で、ニューラルネットワークを用いて、重みを繰り返し更新することで単語分散表現を獲得します。
# 
# 
# ![](./Figure/inference.png)
# 
# ### 推論ベース手法の設計
# 
# 推論ベース手法では、```you 【？】 goodbye and I say hello .```のような、周囲の単語が与えられたときに、```【？】```にどのような単語が出現するのかを推測する推論問題を繰り返し解くことで、単語の出現バターンを学習します。
# 
# つまり、コンテキスト情報を入力として受け取り、各単語の出現する確率を出力する「モデル」を作成することは目標になります。ここで、正しい推測ができるように、コーパスを使って、ニューラルネットワークモデルの学習を行います。そして、その学習の結果として、単語の分散表現を得られます。
# 
# ![](./Figure/inference2.png)
# 
# ### one-hot表現
# 
# ニューラルネットワークで単語を処理するには、それを「固定長のベクトル」に変換する必要があります。
# 
# そのための方法の一つは、単語をone-hot表現へと変換することです。one-hot表現とは、ベクトルの要素の中で一つだけが$1$で、残りは全て$0$であるようなベクトルと言います。
# 
# 単語をone-hot表現に変換するには、語彙数分の要素を持つベクトルを用意して、単語IDの該当する箇所を$1$に、残りは全て$0$に設定します。
# 
# 
# ![](./Figure/one-hot.png)
# 

# ### CBOW（continuous bag-of-words）モデル
# 
# CBOWモデルは、コンテキストからターゲットを推測することを目的としたニューラルネットワークです。このCBOWモデルで、できるだけ正確な推測ができるように訓練することで、単語の分散表現を取得することができます。
# 
# ここで、例として、コンテキスト```["you","goodbye"]```からターゲット```"say"```を予測するタスクを考えます。
# 
# ![](./Figure/cbow.png)

# #### 入力層から中間層(エンコード)
# 
# one-hotエンコーディングで、単語を固定長のベクトルに変換するすることができます。
# 
# 単語をベクトルで表すことができれば、そのベクトルはニューラルネットワークを構成する「レイヤ」によって処理することができるようになりました。
# 
# コンテキストを$\mathbf{c}$、重みを$\mathbf{W}$とし、それぞれ次の形状とします。
# 
# ```{margin}
# イメージしやすいように、ここでは添字を対応する単語で表すことにします。ただし「.(ピリオド)」については「priod」とします。
# ```
# 
# $$
# \mathbf{c}
#     = \begin{pmatrix}
#           c_{\mathrm{you}} & c_{\mathrm{say}} & c_{\mathrm{goodbye}} & c_{\mathrm{and}} & c_{\mathrm{I}} & c_{\mathrm{hello}} & c_{\mathrm{period}}
#       \end{pmatrix}
# ,\ 
# \mathbf{W}
#     = \begin{pmatrix}
#           w_{\mathrm{you},1} & w_{\mathrm{you},2} & w_{\mathrm{you},3} \\
#           w_{\mathrm{say},1} & w_{\mathrm{say},2} & w_{\mathrm{say},3} \\
#           w_{\mathrm{goodbye},1} & w_{\mathrm{goodbye},2} & w_{\mathrm{goodbye},3} \\
#           w_{\mathrm{and},1} & w_{\mathrm{and},2} & w_{\mathrm{and},3} \\
#           w_{\mathrm{I},1} & w_{\mathrm{I},2} & w_{\mathrm{I},3} \\
#           w_{\mathrm{hello},1} & w_{\mathrm{hello},2} & w_{\mathrm{hello},3} \\
#           w_{\mathrm{period},1} & w_{\mathrm{period},2} & w_{\mathrm{period},3} \\
#       \end{pmatrix}
# $$
# 
# コンテキストの要素数(列数)と重みの行数が、単語の種類数に対応します。
# 
# コンテキスト(単語)はone-hot表現として扱うため、例えば「you」の場合は
# 
# $$
# \mathbf{c}_{\mathrm{you}}
#     = \begin{pmatrix}
#           1 & 0 & 0 & 0 & 0 & 0 & 0
#       \end{pmatrix}
# $$
# 
# とすることで、単語「you」を表現できます。
# 
# 重み付き和$\mathbf{h}$は、行列の積で求められます。
# 
# $$
# \begin{aligned}
# \mathbf{h}
#    &= \mathbf{c}_{\mathrm{you}}
#       \mathbf{W}
# \\
#    &= \begin{pmatrix}
#           h_1 & h_2 & h_3
#       \end{pmatrix}
# \end{aligned}
# $$
# 
# $h_1$の計算を詳しく見ると、次のようになります。
# 
# $$
# \begin{aligned}
# h_1
#    &= c_{\mathrm{you}} w_{\mathrm{you},1}
#       + c_{\mathrm{say}} w_{\mathrm{say},1}
#       + c_{\mathrm{goodbye}} w_{\mathrm{goodbye},1}
#       + c_{\mathrm{and}} w_{\mathrm{and},1}
#       + c_{\mathrm{I}} w_{\mathrm{I},1}
#       + c_{\mathrm{hello}} w_{\mathrm{hello},1}
#       + c_{\mathrm{period}} w_{\mathrm{period},1}
# \\
#    &= 1 w_{\mathrm{you},1}
#       + 0 w_{\mathrm{say},1}
#       + 0 w_{\mathrm{goodbye},1}
#       + 0 w_{\mathrm{and},1}
#       + 0 w_{\mathrm{I},1}
#       + 0 w_{\mathrm{hello},1}
#       + 0 w_{\mathrm{period},1}
# \\
#    &= w_{\mathrm{you},1}
# \end{aligned}
# $$
# 
# コンテキストと重みの対応する(同じ単語に関する)要素を掛けて、全ての単語で和をとります。しかしコンテキストは、$c_{you}$以外の要素が$0$なので、対応する重みの値の影響は消えていまします。また$c_{you}$は$1$なので、対応する重みの値$w_{\mathrm{you},1}$がそのまま中間層のニューロンに伝播します。
# 
# 残りの2つの要素も同様に計算できるので、重み付き和
# 
# $$
# \mathbf{h}
#     = \begin{pmatrix}
#           w_{\mathrm{you},1} & w_{\mathrm{you},2} & w_{\mathrm{you},3}
#       \end{pmatrix}
# $$
# 
# は、単語「you」に関する重みの値となります。
# 

# In[1]:


import numpy as np

# 適当にコンテキスト(one-hot表現)を指定
c = np.array([[1, 0, 0, 0, 0, 0, 0]])
print(f"コンテキストの形状：{c.shape}")

# 重みをランダムに生成
W = np.random.randn(7, 3)
print(f"重み\n{W}")

# 重み付き和を計算
h = np.dot(c, W)
print(f"重み付き和\n{h}")
print(f"重み付き和の形状：{h.shape}")


# コンテキストに複数な単語がある場合、入力層も複数になります。このとき、中間層にあるニューロンは、各入力層の全結合による変換後の値が平均されたものになります。
# 
# 中間層のニューロンの数を入力層よりも減らすことによって、中間層には、単語を予測するために必要な情報が"コンパクト"に収められて、結果としては密なベクトル表現が得られます。このとき、この中間層の情報は、人間には理解できない「ブラックボックス」ような状態になります。この作業は、「エンコード」と言います。

# #### 中間層から出力層(デコード)
# 

# 中間層の情報から目的の結果を得る作業は、「デコード」と言います。ここでは、中間層のニューロンの値$\mathbf{h}$を各単語に対応した値になるように、つまり要素(行)数が単語の種類数となるように再度変換したものを、CBOWモデルの出力とします。
# 
# 出力層の重みを
# 
# $$
# \mathbf{W}_{\mathrm{out}}
#     = \begin{pmatrix}
#           w_{1,\mathrm{you}} & w_{1,\mathrm{say}} & w_{1,\mathrm{goodbye}} & w_{1,\mathrm{and}} &
#           w_{1,\mathrm{I}} & w_{1,\mathrm{hello}} & w_{1,\mathrm{period}} \\
#           w_{2,\mathrm{you}} & w_{2,\mathrm{say}} & w_{2,\mathrm{goodbye}} & w_{2,\mathrm{and}} &
#           w_{2,\mathrm{I}} & w_{2,\mathrm{hello}} & w_{2,\mathrm{period}} \\
#           w_{3,\mathrm{you}} & w_{3,\mathrm{say}} & w_{3,\mathrm{goodbye}} & w_{3,\mathrm{and}} &
#           w_{3,\mathrm{I}} & w_{3,\mathrm{hello}} & w_{3\mathrm{period}} \\
#       \end{pmatrix}
# $$
# 
# とします。行数が中間層のニューロン数、列数が単語の種類数になります。
# 
# 出力層も全結合層とすると、最終的な出力は
# 
# $$
# \begin{aligned}
# \mathbf{s}
#    &= \mathbf{h}
#       \mathbf{W}_{\mathrm{out}}
# \\
#    &= \begin{pmatrix}
#           s_{\mathrm{you}} & s_{\mathrm{say}} & s_{\mathrm{goodbye}} & s_{\mathrm{and}} &
#           s_{\mathrm{I}} & s_{\mathrm{hello}} & s_{\mathrm{period}}
#       \end{pmatrix}
# \end{aligned}
# $$
# 

# 例えば、「you」に関する要素の計算は、
# 
# $$
# \begin{aligned}
# s_{\mathrm{you}}
#    &= \frac{1}{2} (w_{\mathrm{you},1} + w_{\mathrm{goodbye},1}) w_{1,\mathrm{you}}
#       + \frac{1}{2} (w_{\mathrm{you},2} + w_{\mathrm{goodbye},2}) w_{2,\mathrm{you}}
#       + \frac{1}{2} (w_{\mathrm{you},3} + w_{\mathrm{goodbye},3}) w_{3,\mathrm{you}}
# \\
#    &= \frac{1}{2}
#       \sum_{i=1}^3
#           (w_{\mathrm{you},i} + w_{\mathrm{goodbye},i}) w_{i,\mathrm{you}}
# \end{aligned}
# 
# $$
# 
# コンテキストに対応する入力層の重みの平均と「you」に関する出力の重みの積になります。
# 
# 他の要素(単語)についても同様に計算できるので、最終的な出力は
# 
# $$
# \begin{aligned}
# \mathbf{s}
#    &= \begin{pmatrix}
#           s_{\mathrm{you}} & s_{\mathrm{say}} & s_{\mathrm{goodbye}} & s_{\mathrm{and}} &
#           s_{\mathrm{I}} & s_{\mathrm{hello}} & s_{\mathrm{period}}
#       \end{pmatrix}
# \\
#    &= \begin{pmatrix}
#           \frac{1}{2} \sum_{i=1}^3 (w_{\mathrm{you},i} + w_{\mathrm{goodbye},i}) w_{i,\mathrm{you}} &
#           \frac{1}{2} \sum_{i=1}^3 (w_{\mathrm{you},i} + w_{\mathrm{goodbye},i}) w_{i,\mathrm{say}} &
#           \cdots &
#           \frac{1}{2} \sum_{i=1}^3 (w_{\mathrm{you},i} + w_{\mathrm{goodbye},i}) w_{i,\mathrm{hello}} &
#           \frac{1}{2} \sum_{i=1}^3 (w_{\mathrm{you},i} + w_{\mathrm{goodbye},i}) w_{i,\mathrm{period}}
#       \end{pmatrix}
# \end{aligned}
# $$
# 
# となります。
# 
# ここで、出力層のニューロンは各単語に対応し、各単語の「スコア」と言います。
# 
# 「スコア」の値が高ければ高いほど、それに対応する単語の出現確率も高くなり、ターゲットの単語であるとして採用します。そのため、スコアを求める処理を推論処理と言います。

# In[2]:


import torch
import torch.nn as nn
import numpy as np

# Define the context data
c0 = torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32) # you
c1 = torch.tensor([[0, 0, 1, 0, 0, 0, 0]], dtype=torch.float32) # goodbye

# Initialize weights randomly
W_in = torch.randn(7, 3, requires_grad=False)  # Input layer weights
W_out = torch.randn(3, 7, requires_grad=False) # Output layer weights

# Define the layers using PyTorch's functional API
def in_layer(x, W):
    return torch.matmul(x, W)

def out_layer(h, W):
    return torch.matmul(h, W)

# Forward pass through the input layers
h0 = in_layer(c0, W_in) # you
h1 = in_layer(c1, W_in) # goodbye
h = 0.5 * (h0 + h1)

# Forward pass through the output layer (scores)
s = out_layer(h, W_out)

# Print the outputs
h0, h1, h, torch.round(s, decimals=3)


# ````{tab-set}
# ```{tab-item} 課題
# 正解は「say」として、Softmax関数によってスコア``s``を確率として扱えるように変換し、そして、正規化した値と教師ラベルを用いて損失を求めなさい。
# ```
# 
# ```{tab-item} ヒント
# 正解は「say」の場合、教師ラベルは``[0, 1, 0, 0, 0, 0, 0]``になります。
# ```
# 
# ````

# #### word2vecの重みと分散表現

# 与えられたコンテキストに対して単語を予測するときに、「良い重み」のネットワークがあれば、「確率」を表すニューロンにおいて、正解に対応するニューロンが高くなっていることが期待できます。そして、大規模コーパスを使って得られる単語の分散表現は、単語の意味や文法のルールにおいて、人間の直感と合致するケースが多く見られます。
# 
# 
# word2vecモデルの学習で行うことが、正しい予測ができるように重みを調整することです。つまり、「コンテキストから出現単語」を予測するという偽タスクをニューラルネットで解いてきましたが、目的はニューラルネットの重みを求めることになります。
# 
# もっと具体的に言えば、word2vecで使用されるネットワークには二つの重みがあります。それは、入力層の重み$\mathbf{W_{in}}$と、出力層の重み$\mathbf{W_{out}}$です。それでは、どちらの重みを使えば良いでしょうか？
# 
# 1. 入力側の重みを利用する
# 2. 出力側の重みを利用する
# 3. 二つの重みの両方を利用する
# 
# Word2Vecモデルに関しては、多くの研究や応用例で、入力層の重みを単語のベクトル表現として使用さており、良好なパフォーマンスを示しています。

# ## Word2Vecモデルの実装

# ### 学習データの準備
# 
# #### コンテキストとターゲット
# 
# Word2Vecモデルためのニューラルネットワークでは、「コンテキスト」を入力した時に、「ターゲット」が出現する確率を高くになるように学習を行います。
# 
# そのため、コーパスから「コンテキスト」と「ターゲット」が対応するデータを作成する必要があります。

# In[3]:


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


# In[4]:


# テキストを設定
text = 'You say goodbye and I say hello.'

# 前処理
corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
print(id_to_word)
print(corpus)


# テキストの単語を単語IDに変換した```corpus```からターゲットを抽出します。
# 
# ターゲットはコンテキストの中央の単語なので、```corpus```の始めと終わりのウインドウサイズ分の単語は含めません。

# In[5]:


# ウインドウサイズを指定
window_size = 1

# ターゲットを抽出
target = corpus[window_size:-window_size]
print(target)


# ターゲットの単語に対して、for文で前後ウィンドウサイズの範囲の単語を順番に抽出し```cs```に格納します。
# 
# つまりウィンドウサイズを$1$とすると、```corpus```におけるターゲットのインデックス```idx```に対して、1つ前(```idx - window_size```)から1つ後(```idx + window_size```)までの範囲の単語を順番に```cs```格納します。ただしターゲット自体の単語はコンテキストに含めません。

# In[6]:


# コンテキストを初期化(受け皿を作成)
contexts = []

# 1つ目のターゲットのインデックス
idx = window_size

# 1つ目のターゲットのコンテキストを初期化(受け皿を作成)
cs = []

# 1つ目のターゲットのコンテキストを1単語ずつ格納
for t in range(-window_size, window_size + 1):
    
    # tがターゲットのインデックスのとき処理しない
    if t == 0:
        continue
    
    # コンテキストを格納
    cs.append(corpus[idx + t])
    print(cs)

# 1つ目のターゲットのコンテキストを格納
contexts.append(cs)
print(contexts)


# In[7]:


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


# In[8]:


# コンテキストとターゲットを作成
contexts, targets = create_contexts_target(corpus, window_size=1)
print(contexts)
print(targets)


# ### PytorchでCBOWモデルの実装
# 

# In[9]:


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# #### Embeddingレイヤ
# 

# 先ほど、理解しやすいone-hot表現でコンテキストを変換する方法を説明しましたが、大規模なコーパスで学習する際、one-hot表現の次元数も大きくになって、非効率な学習の原因になります。
# 
# ただ、one-hot表現による計算は、単に行列の特定の行を抜き出すことだけですから、同じ機能を持つレイヤで入れ替えることは可能です。このような、重みパラメータから「単語IDに該当する行(ベクトル)」を抜き出すためのレイヤは「Embeddingレイヤ」と言います。
# 
# PyTorchで提供されるモジュール```nn.Embedding```を使うと、簡単にEmbeddingレイヤを実装することができます。
# 
# 例えば、語彙に6つの単語があり、各埋め込みベクトルの次元数を3に設定した場合、nn.Embeddingの定義は以下のようになります。
# 
# そして、

# In[10]:


embedding_layer = nn.Embedding(6, 3)


# もしインデックス2のトークンの埋め込みを取得したい場合、次のようにします：

# In[11]:


inputs = torch.tensor([[1,2]], dtype=torch.long)
embedding = embedding_layer(inputs)
embedding


# 埋め込みベクトルの和を取って、入力層から中間層までにエンコードの機能を実装できます。

# In[12]:


out=torch.sum(embedding, dim=1)
out


# In[13]:


linear1 = nn.Linear(3, 6)


# In[14]:


F.log_softmax(linear1(out), dim=1)


# #### ミニバッチ化データセットの作成
# 
# Word2Vecも含めて、深層学習によって学習を行う際には、ミニバッチ化して学習させることが一般的です。
# 
# pytorchで提供されている```DataSet```と```DataLoader```という機能を用いてミニバッチ化を簡単に実現できます。
# 
# ##### DataSet
# 
# DataSetは，元々のデータを全て持っていて、ある番号を指定されると、その番号の入出力のペアをただ一つ返します。クラスを使って実装します。
# 
# DataSetを実装する際には、クラスのメンバ関数として```__len__()```と```__getitem__()```を必ず作ります．
# 
# - ```__len__()```は、```len()```を使ったときに呼ばれる関数です。
# - ```__getitem__()```は、```array[i]```のようにインデックスを使って要素を参照するときに呼ばれる関数です。
# 

# In[15]:


class CBOWDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


# In[16]:


# Convert contexts and targets to tensors
contexts_tensor = torch.tensor(contexts, dtype=torch.long).to(device)
targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)

# Create the dataset
dataset = CBOWDataset(contexts_tensor, targets_tensor)


# In[17]:


print('全データ数:',len(dataset))
print('4番目のデータ:',dataset[3]) 
print('4~5番目のデータ:',dataset[3:5])


# ##### DataLoader
# 
# ```torch.utils.data```モジュールには、データのシャッフとミニバッチの整形に役立つ```DataLoader```というクラスが用意されます。

# In[18]:


# Create the DataLoader
batch_size = 2  # You can adjust the batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[19]:


for data in data_loader:
    print(data)


# #### CBOWモデルの構築

# In[20]:


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# - ```self.embeddings = nn.Embedding(vocab_size, embedding_size)```: 語彙の各単語に対して```embedding_size```次元のベクトルを割り当てる埋め込み層を作成します。
# - ```self.linear1 = nn.Linear(embedding_size, vocab_size)```: 埋め込みベクトルを受け取り、語彙のサイズに対応する出力を生成します。
# - ```embeds = self.embeddings(inputs)```:入力された単語のインデックスに基づいて、埋め込み層から対応するベクトルを取得します。

# In[21]:


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


# In[22]:


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


# ```{margin}
# ``nn.CrossEntropyLoss``ターゲットラベルをクラスのインデックスとして受け取り、内部で必要な変換を行いますので、ターゲットをワンホットエンコーディングに変換する必要はありません。
# ```

# モデルの入力層の重みが単語分散表現であり、$単語 \times  埋め込み次元数$の形の行列になります。

# In[23]:


model.embeddings.weight.shape


# In[24]:


word_embeddings = model.embeddings.weight.data

# 各単語とそれに対応する分散表現を表示
for word, idx in word_to_id.items():
    vector = word_embeddings[idx].cpu().numpy()
    print(f"Word: {word}")
    print(f"Vector: {vector}\n")


# ````{tab-set}
# ```{tab-item} 課題
# 与えられたテキストを用いて、単語分散表現を学習しなさい。
# 
# - ``window_size``を2に設定します
# - ``batch_size``を10に設定します
# 
# ```
# 
# ```{tab-item} テキスト
# 
# "When forty winters shall besiege thy brow,
# And dig deep trenches in thy beauty's field,
# Thy youth's proud livery so gazed on now,
# Will be a totter'd weed of small worth held:
# Then being asked, where all thy beauty lies,
# Where all the treasure of thy lusty days;
# To say, within thine own deep sunken eyes,
# Were an all-eating shame, and thriftless praise.
# How much more praise deserv'd thy beauty's use,
# If thou couldst answer 'This fair child of mine
# Shall sum my count, and make my old excuse,'
# Proving his beauty by succession thine!
# This were to be new made when thou art old,
# And see thy blood warm when thou feel'st it cold."
# 
# ```
# 
# ````

# ## 補足
# 
# ### Negative Sampling

# 今まで紹介した学習の仕組みでは、正例(正しい答え)について学習を行いました。ここで、「良い重み」があれば、ターゲット単語についてSigmoidレイヤの出力は$1$に近づくことになります。
# 
# それだけでなく、本当に行いたいのは、コンテキストが与えられるときに、間違った単語を予測してしまう確率も低いことが望まれます。ただ、全ての誤った単語(負例)を対象として、学習を行うことは非効率であるので、そこで、負例をいくつかピックアップします。これが「Negative Sampling」という手法の意味です。
# 
# Negative Samplingでは、正例をターゲットとした場合の損失を求めると同時に、負例をいくつかサンプリングし、その負例に対しても同様に損失を求めます。そして、両者を足し合わせ、最終的な損失とします。このプロセスは、モデルが正しい単語を予測するだけでなく、不適切な単語を予測しない能力を同時に学習することを目的としています。
# 
# それでは、負例をどのようにサンプリングすべきですか？単純にランダムサンプリングの場合、高頻度の単語はサンプリングされやすく、低頻度の単語はサンプリングされにくく、珍しい単語や文脈に適切に対応できない原因になります。
# 
# この問題点を克服するために、word2vecで提案されるNegative Samplingでは、元となる確率分布に対して以下のように改装しました
# 
# $$
# P'(w_i)= \frac{P(w_i)^{0.75}}{\sum_{i=j}^n P(w_i)^{0.75}}
# $$
# 
# $0.75$乗して調整することで、確率の低い単語に対してその確率を少しだけ高くすることができます。これにより、モデルはより現実的な言語パターンを学習し、実際の使用状況においてより正確な予測を行うことができます。

# ### Skip-gram
# 
# これまで見てきたCBOWモデル以外、word2vecを学習する方法として、Skip-gramと呼ばれる言語モデルが提案されます。
# 
# Skip-gramは、CBOWで扱うコンテキストとターゲットを逆転させて、ターゲットから、周囲の複数ある単語(コンテキスト)を推測します。
# 
# 実に、単語の分散表現の精度の点において、多くの場合、Skip-gramモデルの方が良い結果が得られています。特に、コーパスが大規模になるにつれて、低頻出の単語や類推問題の性能の点において、Skip-gramモデルの方が優れている傾向にあります。

# 

# 
