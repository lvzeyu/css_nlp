#!/usr/bin/env python
# coding: utf-8

# # 単語分散表現
# 
# 単語分散表現とは、単語の意味を低次元の実数値ベクトルで表現することです。
# 
# 機械学習・深層学習モデルは、ベクトル（数値の配列）を入力として受け取ります。テキストを扱う際、最初に決めなければならないのは、文字列を機械学習モデルに入力する前に、数値に変換する（あるいはテキストを「ベクトル化」する）ための戦略です。
# 
# 単語の持つ性質や意味をよく反映するベクトル表現を獲得することは、機械学習・深層学習を自然言語処理で活用するために重要なプロセスです。
# 
# - 類似性: ある概念を表現する際に、ほかの概念との共通点や類似性と紐づけながら、ベクトル空間上に表現します。
# - 単語類推: 分散表現では異なる概念を表現するベクトル同士での計算が可能です
# 
# 
# 
# ![類似性](./Figure/word2vec.png)
# 
# ![単語類推](./Figure/city.png)
# 
# 
# 
# 単語ベクトルへの変換には様々なアプローチが存在します。最初は、(1)人の手によって作られたシソーラス（類語辞書）を利用する手法について簡単の見ていきます。続いて、統計情報から単語を表現する手法ーカウントベースの手法ーについて説明します。この方法は、言語のモデル化を理解することに役に立つと考られます。そして、ニューラルネットワークを用いた手法(具体的には、word2vwcと呼ばれる手法)を扱います。
# 
# 1. シソーラスによる手法
#     - 人の手によって作られたシソーラス（類語辞書）を利用する手法
# 2. カウントベースの手法(今週)
#     - 統計情報から単語を表現する手法
# 3. 推論ベースの手法(来週)
#     - ニューラルネットワークを用いた手法

# ## シソーラスによる手法

# 「単語の意味」を表すためには、人の手によって単語の意味を定義することが考えられます。
# 
# シソーラス(thesaurus)と呼ばれるタイプの辞書は、単語間の関係を異表記・類義語・上位下位といった関係性を用いて、単語間の関連を定義できます。
# 
# $$
# car = auto \ automobile \ machine \ motorcar
# $$
# 
# ![](./Figure/thesaurus.png)
# 
# 
# この「単語ネットーワーク」を利用することで、コンピュータに単語間の関連性を伝えることができます。しかし、この手法には大きな欠点が存在します。
# 
# - 人の作業コストが高い
# - 時代の変化に対応するのが困難
#     - 言語は常に進化しており、新しい単語や意味が生まれては消えていくので、シソーラスを最新の状態に保つのは難しいです。
# - 単語の些細なかニュアンスを表現できない
#     - 単語が持つ複数の意味を区別することは難しい
#     - 単語間の関連性はシソーラスでは静的なものであり、動的な文脈や知識の流れを反映しきれないことがあります

# ## カウントベースの手法

# コーパスには、自然言語に対する人の「知識」ー文章の書き方、単語の選び方、単語の意味ーがふんだんに含まれています。カウントベースの手法の目標は、そのような人の知識が詰まったコーパスから自動的に抽出することにあります。

# ### コーパスの前処理
# 
# コーパスに対して、テキストデータを単語に分割し、その分割した単語をID化にすることが必要されます。
# 
# 単語のID化とは、テキストデータを機械学習モデルなどで処理する際に、単語を一意の整数値（ID）に変換するプロセスを指します。これは、テキストデータをベクトルや行列の形でモデルに入力するための前処理として行われます。
# 

# 例として、簡単なテキストを用意します。

# In[1]:


text = 'You say goodbye and I say hello.'


# In[2]:


# 小文字に変換
text = text.lower()
print(text)

# ピリオドの前にスペースを挿入
text = text.replace('.', ' .')
print(text)

# 単語ごとに分割
words = text.split(' ')
print(words)


# これで、元の文章を単語リストとして利用できるようになりました。これに基づいて、分割した単語と、単語ごとに通し番号を割り振ったIDを2つのディクショナリに格納します。

# In[3]:


# ディクショナリを初期化
word_to_id = {}
id_to_word = {}

# 未収録の単語をディクショナリに格納
for word in words:
    if word not in word_to_id: # 未収録の単語のとき
        # 次の単語のidを取得
        new_id = len(word_to_id)
        
        # 単語IDを格納
        word_to_id[word] = new_id
        
        # 単語を格納
        id_to_word[new_id] = word


# In[4]:


# 単語IDを指定すると単語を返す
print(id_to_word)
print(id_to_word[5])

# 単語を指定すると単語IDを返す
print(word_to_id)
print(word_to_id['hello'])


# 最後に、単語リストから単語IDリストに変換します。

# In[5]:


import numpy as np
# リストに変換
corpus = [word_to_id[word] for word in words]

# NumPy配列に変換
corpus = np.array(corpus)
print(corpus)


# 以上の処理を```preprocess()```という関数として、まとめて実装することにします。

# In[6]:


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


# In[7]:


# テキストを設定
text = 'You say goodbye and I say hello.'

# 単語と単語IDに関する変数を取得
corpus, word_to_id, id_to_word = preprocess(text)
print(id_to_word)
print(word_to_id)
print(corpus)


# ### 分布仮説
# 
# 分布仮説（Distributional Hypothesis）は、言語学や自然言語処理の分野で重要な考え方で、単語の意味は、周囲の単語(コンテキスト)によって形成されるというものです。
# 
# - 単語は、その単語が出現する文脈の集合によって意味が形成されるとされます。同じ文脈で出現する単語は、意味が似ていると考えられます。
# - 単語Aと単語Bが多くの共通の文脈で使用される場合、これらの単語は意味的に関連があると見なされます。
# 
# この仮説は、単語の意味を捉えるためのモデルを作成する際に基本的な原則となっています。
# 
# ![](./Figure/DH.png)

# ### 共起行列
# 
# 分布仮説に基づいた単語ベクトル化の方法を考える際、一番素直な方法は、周囲の単語を"カウント"することです。つまり、ある単語に着目した場合、その周囲どのような単語がどれだけ現れるのかをカウントし、それを集計するのです。
# 
# ここでは、"*You say goodbye and I say hello.*"という文章について、ウィンドウサイズを$1$とする場合、そのコンテキストに含まれる単語の頻度をカウントしてみます。

# ```{figure} ./Figure/co-occur.png
# ---
# align: center
# ---
# 各単語について、そのコンテキストに含まれす単語の頻度
# ```
# 
# 
# - 「You」の周辺単語は「say」のみであり、「say」にのみコンテキストの目印として共起した回数の$1$をカウントします
# - 「say」については文字列中に2回現れていることに注意すると、$[1, 0, 1, 0, 1, 1, 0]$とベクトル表記できます
# 
# 全ての単語に対して、共起する単語をまとめたものを共起行列と呼ばれます。

# In[8]:


# ウィンドウサイズを指定
wndow_size = 1

# 単語の種類数を取得
vocab_size = len(word_to_id)
print(f"単語の種類数: {vocab_size}")

# 総単語数を取得
corpus_size = len(corpus)
print(f"総単語数: {corpus_size}")


# In[9]:


# 共起行列を初期化
co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
print(co_matrix)
print(co_matrix.shape)


# コーパスの7語目の単語「hello」に注目してみます。

# In[10]:


# 単語インデックスを指定
idx = 6

# 指定した単語のIDを取得
word_id = corpus[idx]
print(word_id)
print(id_to_word[word_id])


# In[11]:


# 左隣のインデックス
left_idx = idx - 1
print(left_idx)

# 右隣のインデックス
right_idx = idx + 1
print(right_idx)


# In[12]:


# 左隣の単語IDを取得
left_word_id = corpus[left_idx]
print(left_word_id)
print(id_to_word[left_word_id])

# 共起行列に記録(加算)
co_matrix[word_id, left_word_id] += 1
print(co_matrix)


# In[13]:


# 右隣の単語IDを取得
right_word_id = corpus[right_idx]
print(right_word_id)
print(id_to_word[right_word_id])

# 共起行列に記録(加算)
co_matrix[word_id, right_word_id] += 1
print(co_matrix)

# 対象の単語ベクトル
print(co_matrix[word_id])


# 処理を共起行列を作成する関数として実装します。

# In[14]:


# 共起行列作成関数の実装
def create_co_matrix(corpus, vocab_size, window_size=1):
    
    # 総単語数を取得
    corpus_size = len(corpus)
    
    # 共起行列を初期化
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    # 1語ずつ処理
    for idx, word_id in enumerate(corpus):
        
        # ウィンドウサイズまでの要素を順番に処理
        for i in range(1, window_size + 1):
            # 範囲内のインデックスを計算
            left_idx = idx - i
            right_idx = idx + i
            
            # 左側の単語の処理
            if left_idx >= 0: # 対象の単語が最初の単語でないとき
                # 単語IDを取得
                left_word_id = corpus[left_idx]
                
                # 共起行列にカウント
                co_matrix[word_id, left_word_id] += 1
            
            # 右側の単語の処理
            if right_idx < corpus_size: # 対象の単語が最後の単語でないとき
                # 単語IDを取得
                right_word_id = corpus[right_idx]
                
                # 共起行列にカウント
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix


# 

# 

# 
