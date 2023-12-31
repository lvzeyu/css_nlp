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
contexts, target = create_contexts_target(corpus, window_size=1)
print(contexts)
print(target)


# #### one-hot表現への変換
# 

# 単語IDを要素とするコンテキストとターゲットをone-hot表現のコンテキストとターゲットに変換する関数を実装します。
# 
# 基本的な処理は、単語の種類数個の$0$を要素とするベクトルを作成し、単語ID番目の要素だけを$1$に置き換えます。
# 
# ターゲットは、要素数がターゲット数のベクトルです。変換後は、ターゲット数の行数、単語の種類数の列数の2次元配列になります。つまり、行が各ターゲットの単語、列が各単語IDに対応します。そして行ごとに1つだけ、値が$1$の要素を持ちます。
# 
# -  ```np.zeros()```で変換後の形状の2次元配列を作成し、for文で行ごとに単語ID番目の要素を1を代入します。
# - ```enumerate()```で引数に渡したリストの要素とその要素のインデックスを出力します。
# 

# In[9]:


# ターゲットを確認
print(target)
print(target.shape)

# ターゲットの単語数を取得
N = target.shape[0]

# 単語の種類数を取得
vocab_size = len(word_to_id)

# 全ての要素が0の変換後の形状の2次元配列を作成
one_hot = np.zeros((N, vocab_size), dtype=np.int32)
print(one_hot)

# 単語ID番目の要素を1に置換
for idx, word_id in enumerate(target):
    one_hot[idx, word_id] = 1
print(one_hot)
print(one_hot.shape)


# コンテキストは、0次元目の要素数がターゲット数、1次元目の要素数がウィンドウサイズの$2$倍の2次元配列です。
# 
# - ```np.zeros()```で形状が```(N, C, vocab_size)```である配列を作成し、単語ID番目の要素を1に置換します。

# In[10]:


# コンテキストを確認
print(contexts)
print(contexts.shape)

# ターゲットの単語数を取得
N = contexts.shape[0]

# コンテキストサイズを取得
C = contexts.shape[1]

# 単語の種類数を取得
vocab_size = len(word_to_id)

# 全ての要素が0の変換後の形状の3次元配列を作成
one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
print("全ての要素が0の変換後の形状の3次元配列を作成")
print(one_hot)

# 単語ID番目の要素を1に置換
for idx_0, word_ids in enumerate(contexts): # 0次元方向
    for idx_1, word_id in enumerate(word_ids): # 1次元方向
        one_hot[idx_0, idx_1, word_id] = 1

print("単語ID番目の要素を1に置換")
print(one_hot)
print(one_hot.shape)


# In[11]:


# one-hot表現への変換関数の実装
def convert_one_hot(corpus, vocab_size):
    
    # ターゲットの単語数を取得
    N = corpus.shape[0]
    
    # one-hot表現に変換
    if corpus.ndim == 1: # 1次元配列のとき
        
        # 変換後の形状の2次元配列を作成
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        
        # 単語ID番目の要素を1に置換
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    
    elif corpus.ndim == 2: # 2次元配列のとき
        
        # コンテキストサイズを取得
        C = corpus.shape[1]
        
        # 変換後の形状の3次元配列を作成
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        
        # 単語ID番目の要素を1に置換
        for idx_0, word_ids in enumerate(corpus): # 0次元方向
            for idx_1, word_id in enumerate(word_ids): # 1次元方向
                one_hot[idx_0, idx_1, word_id] = 1
    
    return one_hot


# ### CBOWモデルの実装
# 

# In[12]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCBOW(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size):
        super(SimpleCBOW, self).__init__()
        # 入力層と中間層の重み
        self.in_layer = nn.Embedding(vocab_size, embedding_size)
        # 中間層の次元を調整するための線形層
        self.middle_layer = nn.Linear(embedding_size, hidden_size)
        # 中間層と出力層の重み
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        # 単語の分散表現を初期化
        self.init_weights()

    def init_weights(self):
        # 重みを標準正規分布で初期化
        self.in_layer.weight.data.normal_(0, 1)
        self.middle_layer.weight.data.normal_(0, 1)
        self.out_layer.weight.data.normal_(0, 1)

    def forward(self, contexts):
        # 入力層から中間層への順伝播
        # contextsは周囲の単語のインデックスのバッチ
        embeds = self.in_layer(contexts)  # 埋め込みレイヤーを適用
        h = embeds.mean(dim=1)  # 埋め込みの平均を取る
        h = self.middle_layer(h)  # 中間層に適用
        # 中間層から出力層への順伝播
        out = self.out_layer(h)
        return out

    def loss(self, contexts, target):
        # 順伝播
        out = self.forward(contexts)
        # 損失の計算
        loss = F.cross_entropy(out, target)
        return loss


# In[13]:


model=SimpleCBOW(6, 20, 3)


# In[14]:


model.forward(torch.tensor(contexts))


# In[136]:


# テキストを設定
text = 'You say goodbye and I say hello.'

# 前処理
corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
print(id_to_word)
print(corpus)


# In[95]:


# ウインドウサイズ
window_size = 1

# 単語の種類数を取得
vocab_size = len(word_to_id)
print(vocab_size)

# コンテキストとターゲットを作成
contexts, target = create_contexts_target(corpus, window_size)
print(contexts)
print(contexts.shape)
print(target)
print(target.shape)

# one-hot表現に変換
contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)
print(contexts)
print(contexts.shape)
print(target)
print(target.shape)


# In[126]:


embedding = nn.Embedding(6, 3)


# In[127]:


out=model (torch.tensor(contexts))


# In[130]:


embeds


# In[104]:


h = embeds.mean(dim=1)


# In[107]:


h.shape


# In[55]:


embedding = nn.Embedding(10, 3)


# In[56]:


input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])


# In[123]:


class SimpleCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(SimpleCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)  # 単語埋め込み
        self.linear1 = nn.Linear(embedding_size, hidden_size)  # 中間層
        self.linear2 = nn.Linear(hidden_size, vocab_size)  # 出力層
    
    def forward(self, contexts):
        # contextは [batch_size, context_window * 2]
        embeds = self.embeddings(contexts)  # [batch_size, context_window * 2, embedding_size]
        print(embeds.shape)
        embeds = embeds.sum(dim=1)  # [batch_size, embedding_size]
        print(embeds.shape)
        h = self.linear1(embeds)  # [batch_size, hidden_size]
        h = F.relu(h)  # 活性化関数
        out = self.linear2(h)  # [batch_size, vocab_size]
        return out


# In[132]:


model=SimpleCBOW(6, 20, 3)


# In[133]:


model.forward(torch.tensor(contexts)).shape


# In[134]:


model.forward(torch.tensor(contexts))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
