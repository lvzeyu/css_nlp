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
# 
# 
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

x = torch.rand(5)  # input tensor
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


loss.backward()
print(w.grad)
print(b.grad)


# 最適化ループを構築し、Pytorchより自動的に逆伝播

# In[20]:


import torch.nn.functional as F

def training_loop(n_epochs, learning_rate, model, input, target):
    for epoch in range(1, n_epochs + 1):
        # Forward pass
        outputs = model(input)
        
        # Compute the loss using Binary Cross Entropy with Logits
        loss = F.binary_cross_entropy_with_logits(outputs, target)
        
        # Backward pass
        loss.backward()
        
        # Update the parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        model.zero_grad()
        # Zero the parameter gradients after updating 
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


# In[21]:


# Example usage (with dummy data)
input = torch.rand(10, 512)  # 10 samples with 512 features each
target = torch.rand(10, 3)  # 10 samples with 3 target values each

n_epochs = 500
learning_rate = 0.01
model = NeuralNetwork()

trained_model = training_loop(n_epochs, learning_rate, model, input, target)


# ```{note}
# 
# PyTorchの勾配計算メカニズムでは、``.backward``を呼び出すと、リーフノードで導関数の計算結果が累積されます。つまり、もし``.backward``が以前にも呼び出されていた場合、損失関数が再び計算され、``.backward``も再び呼び出され、各リーフの勾配が前の反復で計算された結果の上に累積されます。その結果、勾配の値は誤ったものになります。
# 
# このようなことが起こらないようにするためには、反復のルーブのたびに``model.zero_grad()``を用いて明示的に勾配をゼロに設定する必要があります。
# 
# 
# ```

# ### 最適化関数
# 
# 最適化は各訓練ステップにおいてモデルの誤差を小さくなるように、モデルパラメータを調整するプロセスです。
# 
# ここまでの説明は、単純な勾配下降法を最適化に使用しました。これは、シンプルなケースでは問題なく機能しますが、モデルが複雑になったときのために、パラメータ学習の収束を助ける最適化の工夫が必要されます。
# 
# #### Optimizer
# 
# ```optim```というモジュールには、様々な最適化アルゴリズムが実装されています。
# 
# ここでは、確率的勾配降下法（Stochastic Gradient Descent）を例として使い方を説明します。
# 
# 確率的勾配降下法は、ランダムに選んだ１つのデータのみで勾配を計算してパラメータを更新し、データの数だけ繰り返す方法です。

# 訓練したいモデルパラメータをoptimizerに登録し、合わせて学習率をハイパーパラメータとして渡すことで初期化を行います。訓練ループ内で、最適化（optimization）は3つのステップから構成されます。
# 
# [1] ``optimizer.zero_grad()``を実行し、モデルパラメータの勾配をリセットします。
# 
# 勾配の計算は蓄積されていくので、毎イテレーション、明示的にリセットします。
# 
# <br>
# 
# [2] 続いて、``loss.backwards()``を実行し、バックプロパゲーションを実行します。
# 
# PyTorchは損失に対する各パラメータの偏微分の値（勾配）を求めます。
# 
# <br>
# 
# [3] 最後に、``optimizer.step()``を実行し、各パラメータの勾配を使用してパラメータの値を調整します。

# In[22]:


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[23]:


def training_loop(n_epochs, learning_rate, model, input, target):
    # Use Binary Cross Entropy with Logits as the loss function
    
    # Use Adam as the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, n_epochs + 1):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input)
        loss = F.binary_cross_entropy_with_logits(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


# In[24]:


input = torch.rand(10, 512)  # 10 samples with 512 features each
target = torch.rand(10, 3)  # 10 samples with 3 target values each

n_epochs = 1000
learning_rate = 0.001
model = NeuralNetwork()

trained_model = training_loop(n_epochs, learning_rate, model, input, target)


# ## 実装例(Irisデータ)

# ### データの読み込み
# 
# Irisデータセットは、アイリス花の3つの異なる種類（Setosa、Versicolour、Virginica）の各50サンプルからなるデータセットです。各サンプルには、以下の4つの特徴値（特徴量）があります。
# 
# - がく片の長さ (sepal length)：アイリス花のがく（緑色の部分）の長さをセンチメートルで測定したもの。
# - がく片の幅 (sepal width)：がくの幅をセンチメートルで測定したもの。
# - 花びらの長さ (petal length)：アイリス花の花びらの長さをセンチメートルで測定したもの。
# - 花びらの幅 (petal width)：花びらの幅をセンチメートルで測定したもの。
# 
# これらの特徴値を使用して、アイリス花の3つの異なる種類を分類することが目標となっています。つまり、目標値（またはラベル）は以下の3つのクラスのいずれかです：
# 
# - Setosa
# - Versicolour
# - Virginica
# このデータセットは、分類アルゴリズムを評価するための基準としてよく使用されます。

# In[25]:


from tensorboardX import SummaryWriter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset and create splits
iris_dataset = datasets.load_iris()


# In[26]:


# Load the iris dataset
iris = datasets.load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier plotting
import pandas as pd
df = pd.DataFrame(data, columns=feature_names)
df['species'] = [target_names[t] for t in target]

# Set a publication-ready theme and increase font scale for better readability
sns.set_theme(style="whitegrid", font_scale=1.2)

# Pair plot to visualize relationships
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue="species", palette="muted", height=2.5, aspect=1.2, plot_kws={'s': 50})
plt.suptitle('Iris Dataset Feature Relationships', y=1.02)

plt.show()


# In[27]:


x_tmp, xtest, y_tmp, ytest = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.2)
xtrain, xval, ytrain, yval = train_test_split(x_tmp, y_tmp, test_size=0.25)  # 0.25 x 0.8 = 0.2 -> 20% validation


# In[28]:


xtrain = torch.from_numpy(xtrain).float()
ytrain = torch.from_numpy(ytrain).long()
xval = torch.from_numpy(xval).float()
yval = torch.from_numpy(yval).long()
xtest = torch.from_numpy(xtest).float()
ytest = torch.from_numpy(ytest).long()


# In[29]:


class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_units, n_out):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(n_in, n_units)
        self.l2 = nn.Linear(n_units, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y


# In[30]:


n_in = xtrain.shape[1]  # number of input features (4 for Iris dataset)
n_units = 10  # number of units in the hidden layer
n_out = 3  # number of classes in the Iris dataset
model = NeuralNetwork(n_in, n_units, n_out)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100


# In[31]:


for epoch in range(n_epochs):
    # Training phase
    model.train()
    outputs = model(xtrain)
    loss = loss_function(outputs, ytrain)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(xval)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == yval).float().mean().item()

    # Print losses and accuracies every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy*100:.2f}%')


# ```{note}
# 
# ``
# writer = SummaryWriter('runs/iris_experiment_1')
# 
# for epoch in range(n_epochs):
#     # Training phase
#     model.train()
#     outputs = model(xtrain)
#     loss = loss_function(outputs, ytrain)
#     
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     
#     writer.add_scalar('Training Loss', loss.item(), epoch)
#     
#     # Validation phase
#     model.eval()
#     with torch.no_grad():
#         val_outputs = model(xval)
#         _, val_predicted = torch.max(val_outputs, 1)
#         val_accuracy = (val_predicted == yval).float().mean().item()
#         writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
# 
#     # Print losses and accuracies every 10 epochs
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy*100:.2f}%')
# 
# ``
# 
# ```

# In[ ]:




