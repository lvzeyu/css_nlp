#!/usr/bin/env python
# coding: utf-8

# # BERTopic
# 
# 社会科学において、定量的な手法によってテキストの内容や構造を明らかにすることは重要な課題である。このような目的を達成するため、トピックモデルがよく用いられます。
# 
# 一般的には、トピックモデルは、テキストデータを構成する各文書の背後にはトピックと呼ばれる語の集合が存在し、それに基づいて各文書が生成されるという仮定から出発します。そして、そのトピックを抽出することにより、テキストデータ全体の傾向を要約することを可能にするモデルです。また、各文書が、どのトピックから生成されているかという点についても示すことが可能となっています。 代表的なモデルとして、Latent Dirichlet allocation (LDA) が挙げられます。
# 
# このような確率的モデルに対して、文章の分散表現を使用してトピックの抽出に文章のコンテキストまで活用しようとする研究が行われています。ここでは、[BERTopic](https://maartengr.github.io/BERTopic/index.html)という手法を解説と実装します
# 

# ## BERTopicの基本概念
# 
# BERTopicのトピック抽出方法は要約すると以下の流れになリます。
# 
# ![](./Figure/berttopic.png)
# 
# - 学習済みモデルで文書の埋め込みを獲得
#    - BERTopicに使われる習済みモデルは、言語モデルの発展に伴って最先端のモデルを利用することもできます
# - 次元削減の手法で文書の埋め込みの次元数を圧縮
#     - 埋め込みベクトルは高次元であるため、より扱いやすい低次元の空間に変換するために次元削減技術を適用します。このステップは、埋め込みベクトルの本質的な特徴を保持しつつ、計算量を減らすことができます
# - 圧縮された文章ベクトルをクラスタリング
#     - 類似したテキストやトピックを共通のグループに分類することができます。
# - (c-TF-IDF による)トピックの代表単語の抽出
#     - 各クラスタに対して、それを最もよく表現する単語やフレーズを選択します。これにより、各クラスタを「トピック」として識別し、それぞれのトピックがどのような内容を含んでいるかを理解することができます。
#     -  BERTopic では、クラスタ内の単語の重要度を知るために TF-IDF を応用してその重要度を算出しています。
#         - クラスタに含まれる文書を全て結合し 1つの文章として扱い、クラスタ単位の TF-IDF (class-based TF-IDF と呼んでいる) を以下のように計算して、それぞれの単語の重要度としています。
#         - クラスタ$c$内の単語$t$のclass-based TF-IDF $W_{t,c}$は以下のようにで得られます
#         
#         $$
#         W_{t, c} = tf_{t,c} \cdot \log \left( 1 + \frac{A}{f_t} \right)
#         $$
# 
#         - $tf_{t,c}$: クラスタ$c$内の単語$t$の単語頻度
#         - $f_t$: 全クラスタに含まれる単語$t$の単語頻度
#         - $A$: クラスタあたりの平均単語数

# In[1]:


#!pip install bertopic


# ## BERTopicの実装

# 
# #### サンプルデータ

# In[2]:


from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']


# In[3]:


docs[:5]


# #### モデルの学習

# - ```language```で学習済みモデルの対応言語を選定します。
#     - ```english```の場合、```all-MiniLM-L6-v2```が使われます。
#     - ```multilingual```、`paraphrase-multilingual-MiniLM-L12-v2`が使われます。
# - ドキュメント集合（`docs`）を使ってトピックモデルを訓練し、ドキュメントごとのトピックとその確率を取得します。

# In[4]:


from bertopic import BERTopic

topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)


# #### トピックの抽出

# In[5]:


freq = topic_model.get_topic_info(); freq.head(5)


# 
# 
# - `Topic`：トピックのID。-1は、特定のトピックに分類されなかったドキュメントを表します。
# - `Count`：そのトピックに分類されたドキュメントの数。
# - `Name`：トピックの名前。これは、そのトピックの主要な単語から生成されます。
# - `Representation`：そのトピックを表現する主要な単語のリスト。
# - `Representative_Docs`：そのトピックに関連する代表的なドキュメントの一部。
# 
# 

# In[6]:


topic_model.get_topic(0)  # Select the most frequent topic


# トピックには、様々な属性を呼び出すメソッドが実装されています。

# | Attribute | Description |
# |------------------------|---------------------------------------------------------------------------------------------|
# | topics_               | The topics that are generated for each document after training or updating the topic model. |
# | probabilities_ | The probabilities that are generated for each document if HDBSCAN is used. |
# | topic_sizes_           | The size of each topic                                                                      |
# | topic_mapper_          | A class for tracking topics and their mappings anytime they are merged/reduced.             |
# | topic_representations_ | The top *n* terms per topic and their respective c-TF-IDF values.                             |
# | c_tf_idf_              | The topic-term matrix as calculated through c-TF-IDF.                                       |
# | topic_labels_          | The default labels for each topic.                                                          |
# | custom_labels_         | Custom labels for each topic as generated through `.set_topic_labels`.                                                               |
# | topic_embeddings_      | The embeddings for each topic if `embedding_model` was used.                                                              |
# | representative_docs_   | The representative documents for each topic if HDBSCAN is used.                                                |

# #### トピックモデルの可視化

# `visualize_topics`メソッドを呼び出すと、各トピックが2次元空間上にプロットされ、類似のトピックが互いに近くに配置されます。これにより、トピック間の関係と各トピックの重要性（サイズによって示される）を視覚的に理解することができます。

# In[7]:


#!pip install --upgrade nbformat
topic_model.visualize_topics()


# トピックの階層関係を確認することができます。

# In[8]:


topic_model.visualize_hierarchy(top_n_topics=50)


# - 特定のトピック内での用語のc-TF-IDFスコアを表します。スコアが高いほど、その用語はそのトピックにとってより関連性が高い、または特徴的であることを示します。
# -  異なるトピックのバーチャートを比較することで、トピックがその主要用語でどのように異なるかを見ることができます。

# In[9]:


topic_model.visualize_barchart(top_n_topics=5)


# ### Topic Representation
# 
# 学習されたトピックモデルの「性能」を向上させるために、追加の最適化ステップが行われます。
# 
# - `update_topics`メソッドで、既存のトピックモデルが指定するドキュメント集合（`docs`）に基づいて更新されます。ここでは、`n_gram_range=(1, 2)`と指定し、モデルが考慮する単語の組み合わせの範囲を1から2に設定することを意味します。つまり、単語単体（1-gram）と単語のペア（2-gram）が考慮されます。
# 

# In[10]:


topic_model.update_topics(docs, n_gram_range=(1, 2))


# In[11]:


topic_model.get_topic(0)   # We select topic that we viewed before


# - `reduce_topics`でトピックモデルのトピック数を減らします。トピックモデルが生成するトピックの数を制御するための方法で、トピックの数が多すぎて解釈が難しい場合や、トピック間の区別が不明瞭な場合に有用です。

# In[12]:


topic_model.reduce_topics(docs, nr_topics=60)


# - `find_topics`メソッドで、指定したキーワードに最も関連性の高いトピックを検索します。このメソッドを呼び出すと、指定したキーワードに最も関連性の高いトピックのIDとその関連性のスコアが返されます。

# In[13]:


similar_topics, similarity = topic_model.find_topics("vehicle", top_n=5); similar_topics


# In[20]:


topic_model.get_topic(3)

