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

# 
