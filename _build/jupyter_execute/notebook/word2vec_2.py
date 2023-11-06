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
# 正解は「you」として、Softmax関数によってスコア``s``を確率として扱えるように変換し、そして、正規化した値と教師ラベルを用いて損失を求めなさい。
# ```
# 
# ```{tab-item} ヒント
# 正解は「you」の場合、教師ラベルは``[0, 1, 0, 0, 0, 0, 0]``になります。
# ```
# 
# ````

# ### CBOWモデルの学習
# 

# In[3]:


torch.softmax(s, dim=1)


# In[4]:


t = torch.tensor([[0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
loss = nn.CrossEntropyLoss()
loss(s,t)


# 
# t = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
# 
# loss()
