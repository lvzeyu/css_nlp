#!/usr/bin/env python
# coding: utf-8

# # LSTMの実装
# 
# ## [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

# PyTorchの[torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)クラスを使用して、LSTMをモデルに簡単にレイヤーとして追加できます。

# 
# `torch.nn.LSTM`の主なパラメータは以下の通りです：
# 
# - `input_size`：入力xの特徴量の数
# - `hidden_size`：隠れ状態の特徴量の数
# - `num_layers`：LSTMを重ねる層の数（デフォルトは$1$）
# - `bias`：バイアスを使用するかどうか（デフォルトは`True`）
# - `batch_first`：入力と出力のテンソルの形状が(`batch`, `seq`, `feature`)になるようにするかどうか（デフォルトは`False`）
# - `dropout`：ドロップアウトを適用する確率（デフォルトは$0$、つまりドロップアウトなし）
# - `bidirectional`：双方向LSTMを使用するかどうか（デフォルトは`False`）

# ## 入力

# - `input_size=10`：各入力要素の特徴の数（入力ベクトルの次元数）は10です
# - `hidden_size=20`：隠れ状態とセル状態の各ベクトルのサイズは20です
# - `num_layers=2`： LSTMの層の数は2です
#     - 最初のLSTM層は入力シーケンスを受け取り、それを処理して一連の隠れ状態を生成します。
#     - 最初の層の出力（隠れ状態）は、次のLSTM層の入力となります。これが複数層にわたって繰り返されます。

# In[1]:


import torch
import torch.nn as nn
# LSTMのインスタンス化
lstm = nn.LSTM(input_size=10, hidden_size=20,num_layers=2)


# - 入力データの生成（例：バッチサイズ=3, シーケンス長=5）

# In[2]:


input = torch.randn(5, 3, 10)


# - `h0`:隠れ状態の初期値
#     - `h0` のサイズは`(num_layers * num_directions, batch_size, hidden_size)`になります
#         - `num_directions`:LSTMが単方向か双方向かを示し（単方向の場合は$1$、双方向の場合は$2$）
# - `c0`:セル状態の初期値
#     - `c0` のサイズは同様に`(num_layers * num_directions, batch_size, hidden_size)`になります

# In[3]:


# 隠れ状態とセル状態の初期化
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)


# ## 出力
# 
# `torch.nn.LSTM`の出力は、出力テンソル（通常は `output` と呼ばれます）と隠れ状態（`h_n` と `c_n`）から構成されています

# In[4]:


# 順伝播
output, (hn, cn) = lstm(input, (h0, c0))


# In[5]:


print(f"output: {output.shape}")
print(f"hn: {hn.shape}")
print(f"cn: {cn.shape}")


# - 出力テンソル（`output`）
#     - シーケンス内の各時点におけるLSTMの隠れ状態を含んでいます。
#     - サイズは `(seq_len, batch, num_directions * hidden_size)` になります。
# - 最終隠れ状態(`h_n`)
#     - LSTMの最終的な隠れ状態です。
#     - サイズは`(num_layers * num_directions, batch, hidden_size)` になります。
# 
# ```{note}
# - outputの最終時点の隠れ状態（つまり output[-1] ）は、単層、単方向LSTMの場合、hn と同じです。
# - 多層LSTMの場合、hnは各層の最終隠れ状態を含むため、outputの最終時点の隠れ状態とは異なることがあります。この場合、outputの最後の要素は最終層の最終隠れ状態に対応し、hn にはそれぞれの層の最終隠れ状態が格納されます。
# ```
# 
# - 最終セル状態(`c_n`)
#     - LSTMの最終的なセル状態です。長期的な依存関係をどのように「記憶」しているかを示します。
#     - サイズは `(num_layers * num_directions, batch, hidden_size)` です
#    

# ````{tab-set}
# ```{tab-item} 質問1
# タスクに応じて、torch.nn.LSTMの出力を扱う必要があります。例えば、テキスト生成、機械通訳、文書分類はそれぞれどの出力を使うべきですか？
# ```
# ````

# 
