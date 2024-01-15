#!/usr/bin/env python
# coding: utf-8

# # GPT

# GPT(Generative Pretrained Transformer)はTransformerベースの言語モデルです。ChatGPT などの生成系 AI アプリケーションの基礎となっている人工知能 (AI) の重要な新技術です。GPT モデルにより、アプリケーションは人間のようにテキストやコンテンツ (画像、音楽など) を作成したり、会話形式で質問に答えたりすることができます。さまざまな業界の組織が、Q&A ボット、テキスト要約、コンテンツ生成、検索に GPT モデルと生成系 AI を使用しています。
# 
# GPTはOpenAIによって定期的に新しいバージョンが公開されていますが、ここではGPT-2について解説します。
# 
# ![](./Figure/gpt_history.png)
# 
# ![](./Figure/gpt2.png)
# 

# ## 入力表現

# GPTの入力は、入力トークン列に対応するトークン埋め込み$e_{w_i}$と位置埋め込む$p_i$を加算した埋め込み列です。
# 
# $$
# x_i=e_{w_i}+p_i
# $$

# ## 事前学習

# GPTの事前学習タスクは、入力されたトークン列の次のトークンを予測することです。ここで、GPTはデコーダ構成のTransformerを用います。
# 
# ![](./Figure/gpt_input_representation.png)
# 
# ```{margin}
# GPTはオリジナルのTransformerにいくつかの改装を行いました。
# ```
# 
# 学習に用いるトークン列$w_1,w_2,...,w_N$におけるのトークン$w_i$を予測することを考えます。GPTでは、予測確率を使った負の対数尤度を損失関数として事前学習を行います。
# 
# $$
# \zeta(\theta)=- \sum_i log P(w_i|w_{i-K},....w_{i-1},\theta)
# $$
# 
# ここで、$\theta$はモデルに含まれるすべてのパラメータを表します。
# 
# 学習時にはMasked Self-Attention機構が導入され、入力トークン列の各位置において次のトークンを予測して学習が行われます。
# 
# 
# ![](./Figure/gpt_decoder.png)
# 
# 学習時にはMasked Self-Attention機構が導入され、入力トークン列の各位置において次のトークンを予測して学習が行われます。
# 

# ## ファインチューング
# 
# GPTの事前学習済みモデルに、下流タスクに合わせて変換するためのヘッドを追加し、下流タスクのデータセットを用いてモデル全体を調整します。
# 
# GPTは下流タスクを解く際、特殊トークンを用いて入力テキストを拡張します。
# 
# - テキスト分類のタスクにおいては、文書の最初に```<s>```、最後に```<e>```が追加されます。
# - 自然言語推論のタスクにおいては、テキストの境界に```$```が挿入されます。
# 
# ![](./Figure/gpt-2-autoregression-2.gif)
# 

# ### Huggingface transformerを使う
# 

# In[1]:


from transformers import pipeline
#!pip install sentencepiece
#!pip install protobuf
generator = pipeline("text-generation", model="abeja/gpt2-large-japanese")


# In[2]:


generated = generator(
    "東北大学は",
    max_length=100,
    do_sample=True,
    num_return_sequences=3,
    top_p=0.95,
    top_k=50,
    pad_token_id=3
)
print(*generated, sep="\n")

