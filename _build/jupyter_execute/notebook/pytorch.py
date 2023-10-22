#!/usr/bin/env python
# coding: utf-8

# # Pytorch

# [PyTorch](https://pytorch.org/)はPythonのオープンソースの機械学習・深層学習ライブラリです。
# 
# - 柔軟性を重視した設計であり、さらに、機械学習・深層学習モデルをPythonの慣用的なクラスや関数の取り扱い方で実装できるようになっています。
# - GPUを使用した計算をサポートしますので、CPU上で同じ計算を行う場合に比べて、数十倍の高速化を実現します。

# In[1]:


#pip install torch torchvision torchaudio
import torch


# ## テンソル
# 
# 深層学習モデルは通常、入力から出力にどのようにマッピングされるのかを対応つけるデータ構造を表します。一般的に、このようなある形式のデータから別の形式への変換は膨大な浮動小数点数の計算を通じて実現されています。
# 
# データを浮動小数点数を扱うためには、Pytorchは基本的なデータ構造として「テンソル」を導入しています。
# 
# 深層学習の文脈でのテンソルとは、ベクトルや行列を任意の次元数に一般化したものを指します。つまり、多次元配列を扱います。
# 
# ```{margin}
# Tensorとの同じように、NumPyも多次元配列を扱えます。ただ、PyTorchにおいてテンソルはGPU上でも使用できるため、処理速度の向上させることも可能です。
# ```
# 
# ![](/Users/ryozawau/css_nlp/notebook/Figure/tensor.png)

# ### テンソルの作成

# In[2]:


x = torch.ones(5, 3)
print(x)


# In[3]:


x = torch.rand(5, 3)
print(x)


# In[4]:


x = torch.tensor([5.5, 3])
print(x)


# ### テンソル要素の型
# 
# テンソル要素の型は、引数に適切な```dtype```を渡すことで指定します。

# In[5]:


double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)


# ### テンソルの操作（変形・変換等）
# 
# PyTorchにはテンソルに対する[操作（変形・演算など）](https://torch7.readthedocs.io/en/rtd/maths/index.html)が多く用意されています。
# 

# In[6]:


x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)


# In[7]:


print(torch.add(x, y))


# ### テンソルの一部指定や取り出し(Indexing)
# 
# Pytorchテンソルは、Numpyや他のPythonの科学計算ライブラリーと同じく、テンソルの次元ごとのレンジインデックス記法で一部指定や取り出しを行えます。

# In[8]:


x[3:,:]


# In[9]:


x[1:,0]


# ### CUDA Tensors（CUDA テンソル）
# 
# tensorは ```.to``` メソッドを使用することであらゆるデバイス上のメモリへと移動させることができます。

# In[10]:


if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


# ## モデル構築

# [`torch.nn`](https://pytorch.org/docs/stable/nn.html)で用意されているクラス、関数は、独自のニューラルネットワークを構築するために必要な要素を網羅しています。
# 
# PyTorchの全てのモジュールは、[`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)を継承しています。
# 
# 
# そしてニューラルネットワークは、モジュール自体が他のモジュール（レイヤー）から構成されています。
# 
# この入れ子構造により、複雑なアーキテクチャを容易に構築・管理することができます。

# ### クラスの定義
# 
# ``nn.Module``を継承し、独自のネットワークモデルを定義し、その後ネットワークのレイヤーを ``__init__``で初期化します。
# 
# ``nn.Module`` を継承した全モジュールは、入力データの順伝搬関数である``forward``関数を持ちます。
# 

# In[11]:


from torch import nn


# In[12]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# このクラスは、PyTorchの```nn.Module```を継承した単純なニューラルネットワークの実装を示しています。入力は固定長の$512$とされており、出力は$3$の次元を持つベクトルです。
# 
# ```{margin}
# 最大長512であるテキストに対して、センチメント(ポジティブ、中立、ネガティブ)を予測するタスクをイメージしてください。
# ```
# 
# - ```self.linear_relu_stack```: このシーケンシャルな層は、3つの線形層とそれぞれの後に続くReLU活性化関数から構成されています。
#     - [`linear layer`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)は、線形変換を施します。`linear layer`は重みとバイアスのパラメータを保持しています。
#     - [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)という活性化関数を設置することで、ニューラルネットワークの表現力を向上させます。
# - 順伝播メソッド (```forward```): 入力テンソル```x```を受け取り、ネットワークを通して出力を生成する機能を持ちます。

# ``NeuralNetwork``クラスのインスタンスを作成し、変数``device``上に移動させます。
# 
# 以下でネットワークの構造を出力し確認します。

# In[13]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# In[14]:


model = NeuralNetwork().to(device)
print(model)


# - ニューラルネットワークの最後のlinear layerは`logits`を出力します。この`logits`は[`nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)モジュールへと渡されます。出力ベクトルの要素の値は$[0, 1]$の範囲となり、これは各クラスである確率を示します。

# In[15]:


X = torch.rand(3, 512, device=device)
logits = model(X) 
print(logits)


# In[16]:


pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# ### クラスのレイヤー
# 

# ```{note}
# tensorboardでニューラルネットワークの構造を確認する
# ``
# from torch.utils.tensorboard import SummaryWriter
# X = torch.rand(3, 28, 28)
# writer = SummaryWriter("torchlogs/")
# writer.add_graph(model, X)
# writer.close()
# ``
# 
# ```

# ## 自動微分
# 
# ニューラルネットワークを訓練する際、その学習アルゴリズムとして、**バックプロパゲーション（back propagation）** がよく使用されます。
# 
# バックプロパゲーションでは、モデルの重みなどの各パラメータは、損失関数に対するその変数の微分値（勾配）に応じて調整されます。
# 
# これらの勾配の値を計算するために、PyTorchには``torch.autograd`` という微分エンジンが組み込まれています。
# 
# autogradはPyTorchの計算グラフに対する勾配の自動計算を支援します。
# 
# シンプルな1レイヤーのネットワークを想定しましょう。
# 
# 入力を``x``、パラメータを``w`` と ``b``、そして適切な損失関数を決めます。
# 
# <br>
# 
# PyTorchでは例えば以下のように実装します。

# In[17]:


import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


# ### 勾配情報の保存

# こののニューラルネットワークでは、``w``と``b``が最適したいパラメータです。
# 
# そのため、これらの変数に対する損失関数の微分値を計算する必要があります。
# 
# これらのパラメータで微分を可能にするために、``requires_grad``属性をこれらのテンソルに追記します。
# 
# そうすると、勾配は、テンソルの ``grad_fn`` プロパティに格納されます。
# 
# 

# In[18]:


print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)


# ### 勾配計算
# 
# ニューラルネットワークの各パラメータを最適化するために、入力``x``と出力``y``が与えられたもとで、損失関数の各変数の偏微分値、
# 
# すなわち
# 
# $\frac{\partial loss}{\partial w}$ 、$\frac{\partial loss}{\partial b}$ 
# 
# を求める必要があります。
# 
# 
# これらの偏微分値を求めるために``loss.backward()``を実行し、``w.grad``と``b.grad``の値を導出します。
# 
# 逆伝搬では、``.backward()``がテンソルに対して実行されると、autogradは、
# - 各変数の ``.grad_fn``を計算する
# - 各変数の``.grad``属性に微分値を代入する
# - 微分の連鎖律を使用して、各leafのテンソルの微分値を求める
# 
# を行います。

# In[19]:


loss.backward(retain_graph=True)
print(w.grad)
print(b.grad)


# 

# 
