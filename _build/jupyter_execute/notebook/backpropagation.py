#!/usr/bin/env python
# coding: utf-8

# # 誤差逆伝播法
# 
# これまでは、ニューラルネットワークの各パラメータについての目的関数の数値微分を計算することで勾配の計算を求める方法を説明しました。
# 
# しかし、ニューラルネットワークの層数が多くなると、数値微分の計算は膨大な時間がかかるでしょう。
# 
# ここで、パラメータの勾配の計算を効率よく行う手法である「誤差逆伝播法」について学びます。
# 
# ![誤差逆伝播法 (backpropagation) の計算過程](https://tutorials.chainer.org/ja/_images/13_backpropagation.gif)

# ## 連鎖律

# 複数の関数によって構成される関数を合成関数と呼びます。
# 
# $$
# \begin{align}
# z &= t^2 \\
# t &= x + y
# \end{align}
# $$
# 
# 合成関数の微分は、「$t$に関する$z$の微分$\frac{\partial z}{\partial t}$」と「$x$に関する$t$の微分$\frac{\partial t}{\partial 1}$」の積のように、それぞれの関数の微分の積で求められます。
# 
# $$
# \frac{\partial z}{\partial x}
#     = \frac{\partial z}{\partial t}
#       \frac{\partial t}{\partial x}
# $$

# ## 線形変換の逆伝播の導出
# 
# 入力データ$\mathbf{x}$は$(N \times D)$の行列、$\mathbf{W}$は$(D \times H)$の行列、$\mathbf{b}$は要素数$H$のベクトルと考え、線形変換の計算は以下の式で表します。

# $$
# \begin{aligned}
# \mathbf{y}
#    &= \mathbf{x} \mathbf{W} + \mathbf{b}
# \\
#    &= \begin{pmatrix}
#           x_{0,0} & x_{0,1} & \cdots & x_{0,D-1} \\
#           x_{1,0} & x_{1,1} & \cdots & x_{1,D-1} \\
#           \vdots & \vdots & \ddots & \vdots \\
#           x_{N-1,0} & x_{N-1,1} & \cdots & x_{N-1,D-1}
#       \end{pmatrix}
#       \begin{pmatrix}
#           w_{0,0} & w_{0,1} & \cdots & w_{0,H-1} \\
#           w_{1,0} & w_{1,1} & \cdots & w_{1,H-1} \\
#           \vdots & \vdots & \ddots & \vdots \\
#           w_{D-1,0} & w_{D-1,1} & \cdots & w_{D-1,H-1}
#       \end{pmatrix}
#       + \begin{pmatrix}
#           b_0 & b_1 & \cdots & b_{H-1}
#         \end{pmatrix}
# \\
#    &= \begin{pmatrix}
#           \sum_{d=0}^{D-1} x_{0,d} w_{d,0} + b_0 & 
#           \sum_{d=0}^{D-1} x_{0,d} w_{d,1} + b_1 & 
#           \cdots & 
#           \sum_{d=0}^{D-1} x_{0,d} w_{d,H-1} + b_{H-1} \\
#           \sum_{d=0}^{D-1} x_{1,d} w_{d,0} + b_0 & 
#           \sum_{d=0}^{D-1} x_{1,d} w_{d,1} + b_1 & 
#           \cdots & 
#           \sum_{d=0}^{D-1} x_{1,d} w_{d,H-1} + b_{H-1}  \\
#           \vdots & \vdots & \ddots & \vdots \\
#           \sum_{d=0}^{D-1} x_{N-1,d} w_{d,0} + b_0 & 
#           \sum_{d=0}^{D-1} x_{N-1,d} w_{d,1} + b_1 & 
#           \cdots & 
#           \sum_{d=0}^{D-1} x_{N-1,d} w_{d,H-1} + b_{H-1} 
#       \end{pmatrix}
# \\
#    &= \begin{pmatrix}
#           y_{0,0} & y_{0,1} & \cdots & y_{0,H-1} \\
#           y_{1,0} & y_{1,1} & \cdots & y_{1,H-1} \\
#           \vdots & \vdots & \ddots & \vdots \\
#           y_{N-1,0} & y_{N-1,1} & \cdots & y_{N-1,H-1}
#       \end{pmatrix}
# \end{aligned}
# 
# $$

# ここで、「$n$番目の出力データの$h$番目の項$y_{n,h}$」は、
# 
# $$
# y_{n,h}
#     = \sum_{d=0}^{D-1} x_{n,d} w_{d,h} + b_h
# $$
# 
# で計算できるのが分かります。
# 

# ### 重みの勾配

# 連鎖律より、$\frac{\partial L}{\partial w_{d,h}}$は次の式で求められます
# 
# $$
# \frac{\partial L}{\partial w_{d,h}}
#     = \sum_{n=0}^{N-1}
#           \frac{\partial L}{\partial y_{n,h}}
#           \frac{\partial y_{n,h}}{\partial w_{d,h}}
# $$
# 
# - $\frac{\partial L}{\partial y_{n,h}}$は、$y_{n,h}$に関する$L$の微分です。
# - $\frac{\partial y_{n,h}}{\partial w_{d,h}}$は、$w_{d,h}$に関する$y_{n,h}$の微分です。
# 
# ここで、$\frac{\partial y_{n,h}}{\partial w_{d,h}}$は、

# $$
# \begin{aligned}
# \frac{\partial y_{n,h}}{\partial w_{d,h}}
#    &= \frac{\partial}{\partial w_{d,h}} \left\{
#           \sum_{d=0}^{D-1} x_{n,d} w_{d,h} + b_h
#       \right\}
# \\
#    &= \frac{\partial}{\partial x_{n,d}} \Bigl\{
#           x_{n,0} w_{0,h} + \cdots + x_{n,d} w_{d,h} + \cdots + x_{n,D-1} w_{D-1,h} + b_h
#       \Bigr\}
# \\
#    &= 0 + \cdots + x_{n,d} + \cdots + 0 + 0
# \\
#    &= x_{n,d}
# \end{aligned}
# $$
# 
# になりますため、
# 
# $$
# \frac{\partial L}{\partial w_{d,h}}
#     = \sum_{n=0}^{N-1}
#           \frac{\partial L}{\partial y_{n,h}}
#           x_{n,d}
# $$

# ### バイアスの勾配
# 
# 同じく連鎖律より、$\frac{\partial L}{\partial b_h}$は次の式で求められます。
# 
# $$
# \frac{\partial L}{\partial b_h}
#     = \sum_{n=0}^{N-1}
#           \frac{\partial L}{\partial y_{n,h}}
#           \frac{\partial y_{n,h}}{\partial b_h}
# $$
# 
# $$
# \begin{aligned}
# \frac{\partial y_{n,h}}{\partial b_h}
#    &= \frac{\partial}{\partial w_{d,h}} \left\{
#           \sum_{d=0}^{D-1} x_{n,d} w_{d,h} + b_h
#       \right\}
# \\
#    &= 0 + 1
# \\
#    &= 1
# \end{aligned}
# $$
# 
# まとめると、
# 
# $$
# \frac{\partial L}{\partial b_h}
#     = \sum_{n=0}^{N-1}
#           \frac{\partial L}{\partial y_{n,h}}
# 
# $$

# ## ニューラルネットワークにおける誤差逆伝播法
# 
# 連鎖律より勾配を計算する考え方をニューラルネットワークにも適用することができます。具体的には、ニューラルネットワークを構成する関数が持つパラメータについての**目的関数の勾配**を、順伝播で通った経路を逆向きにたどるようにして**途中の関数の勾配の掛け算**によって求めます。
# 
# ```{margin}
# ニューラルネットワークには、活性化関数によて変換し、次の層へ伝播するといった計算の流れになりますが、逆伝播による勾配を計算できる原理は変わらないです。
# ```

# ここから、手計算を通じて誤差逆伝播法の実装を理解しましよう。
# 
# - 入力
# 
# $$
# i_{1} = 0.05,i_{2} = 0.10
# 
# $$
# - 初期パラメータ
# 
# $$
# w_{1} = 0.15,w_{2} = 0.20,w_{3} = 0.25,w_{4} = 0.30
# $$
# 
# $$
# w_{5} = 0.40,w_{6} = 0.45,w_{7} = 0.50,w_{8} = 0.55
# $$
# 
# - 活性化関数: シグモイド関数
# 
# $$
# h(x)
#     = \frac{1}{1 + \exp(-x)}
# $$
# 
# - 教師データ
# $$
# o_{1} = 0.01,o_{2} = 0.99
# $$
# 
# - 目的関数は平均二乗誤差関数を用いることにします。
# 
# $$
# L = \dfrac{1}{N} \sum_{n=1}^{N} (t_{n} - y_{n})^2
# $$
# 
# 
# ```{figure} ./Figure/back1.png
# ---
# align: center
# ---
# ニューラルネットワークの実装例
# ```
# 
# 

# ### 順伝播の流れ
# 

# In[1]:


import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[2]:


net_h1= (0.15)*(0.05)+(0.2)*(0.1)+0.35
print("net_h1={}".format(net_h1))


# In[3]:


net_h2= (0.25)*(0.05)+(0.3)*(0.1)+0.35
print("out_h2={}".format(net_h2))


# In[4]:


net_o1 = (0.4)*net_h1+(0.45)*net_h2+0.6
out_o1= sigmoid(net_o1)
print("out_o1={}".format(out_o1))
net_o2 = (0.5)*net_h1+(0.55)*net_h2+0.6
out_o2= sigmoid(net_o2)
print("out_o2={}".format(out_o2))


# In[5]:


L_1 = 0.5 * np.square(0.01-out_o1)
L_2 = 0.5 * np.square(0.99-out_o2)
L = L_1+L_2
print("Loss={}".format(L))


# 例えば、$w_5$の勾配を計算する際には、
# 
# ```{figure} ./Figure/back2.png
# ---
# align: center
# ---
# 誤差逆伝播法で$w_5$の勾配を求める
# ```
# 
# 
# $$
# \frac{\partial L}{\partial w_5} = \frac{\partial L}{\partial out_{o1}}\frac{\partial out_{o1}}{\partial net_{o1}}\frac{\partial net_{o1}}{\partial w_5}
# $$

# $\frac{\partial L}{\partial out_{o1}}$を計算する
# 
# $$
# L= \frac{1}{2}(target_{o_{1}}-out_{o_{1}})^2+\frac{1}{2}(target_{o_{2}}-out_{o_{2}})^2
# $$
# 
# 合成関数の微分$g(f(x))= g^{\prime}(f(x))f^{\prime}(x)$によって
# 
# $$
# \frac{\partial L}{\partial out_{o1}}= 2*\frac{1}{2}(target_{o_{1}}-out_{o_{1}})*-1+0
# $$

# In[6]:


d_out_o1 = -(0.01-out_o1)
print("d_out_o1={}".format(d_out_o1))


# $\frac{\partial out_{o1}}{\partial net_{o1}}$を計算する
# 
# $$
# out_{o1}= sigmod(net_{o_{1}})
# $$
# 
# Sigmoid関数の微分は $f^{\prime}(x)=f(x)(1-f(x))$ なので
# 
# $$
# \frac{\partial out_{o1}}{\partial net_{o1}}= out_{o1}(1-out_{o1})
# $$
# 

# ```{note}
# 
# シグモイド関数の勾配の証明
# 
# $$
# \begin{aligned}
# \frac{d y}{d x}
#    &= \frac{d}{d x} \Bigl\{
#           \frac{1}{1 + \exp(-x)}
#       \Bigr\}
# \\
#    &= - \frac{1}{(1 + \exp(-x))^2}
#         \frac{d}{d x} \Bigl\{
#             1 + \exp(-x)
#         \Bigr\}
# \\
#    &= - \frac{1}{(1 + \exp(-x))^2} \Bigl(
#             - \exp(-x)
#         \Bigr)
# \\
#    &= \frac{\exp(-x)}{(1 + \exp(-x))^2}
# \\
#   &= \frac{1}{1 + \exp(-x)}
#       \frac{\exp(-x)}{1 + \exp(-x)}
# \\
#    &= \frac{1}{1 + \exp(-x)}
#       \frac{1 + \exp(-x) - 1}{1 + \exp(-x)}
# \\
#    &= \frac{1}{1 + \exp(-x)} \left(
#           \frac{1 + \exp(-x)}{1 + \exp(-x)}
#           - \frac{1}{1 + \exp(-x)}
#       \right)
# \\
#    &= y (1 - y)
#     \end{aligned}
# $$
# 
# ```

# In[7]:


d_net_o1 = out_o1*(1-out_o1)
print("d_net_o1={}".format(d_net_o1))


# $\frac{\partial net_{o1}}{\partial w_5}$を計算する
# 
# $$
# net_{o_{1}}=w_{5}*net_{h_{1}}+w_{6}*net_{h_{2}}+b_{2}*1
# $$
# 
# $$
# \frac{\partial net_{o1}}{\partial w_5}= net_{h_{1}}
# $$

# In[8]:


d_w5= d_out_o1*d_net_o1*net_h1
print("d_w5={}".format(d_w5))


# パラメータを更新する
# 
# $$
# w_5^+ = w_{5}- \eta \frac{\partial {L}}{\partial w_5}
# $$
