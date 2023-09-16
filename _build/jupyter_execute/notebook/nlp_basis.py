#!/usr/bin/env python
# coding: utf-8

# # 自然言語処理の基礎

# ## コーパス 
# 
# ### コーパスとは
# 
# 「コーパス（Corpus）」とは、自然言語の文章や使い方を大規模に収集し、コンピュータで検索できるよう整理されたデータのことです。
# 
# コーパスが大きく「ラベル付きコーパス」と「ラベルなしコーパス」に分けられます。
# 
# - ラベルなしコーパス: 注釈付けを行わずにテキストを集めただけのコーパス。例えば、スクリーンスクレイピングによる取得するウェブベージの生テキストデータ。
# - ラベル付きコーパス: ラベル付きコーパスとは、テキストに付加情報（ラベルや注釈）が付与されているコーパスのことを指します。このラベルは、特定のタスクや研究の目的に応じてデータに追加されます。
#     - 品詞タグ付きコーパス: 各単語に品詞（名詞、動詞、形容詞など, part of speech）のタグが付けられています。これは、文法的構造の理解や、品詞タガーの訓練データとして使用されます。
#     - 固有名詞抽出のためのコーパス: テキスト内の固有名詞（人名、地名、組織名など）に対して、該当するラベルが付けられています。
# 
# ```
# <sentence>
# 　<LUW B="B" SL="v" l_lemma="公共工事請け負い金額" l_lForm="コウキョウコウジウケオイキンガク" l_wType="混" l_pos="名詞-普通名詞-一般" >
# 　　<SUW lemma="公共" lForm="コウキョウ" wType="漢" pos="名詞-普通名詞-一般" pron="コーキョー">
# 　　　公共
# 　　</SUW>
# 　　<SUW lemma="工事" lForm="コウジ" wType="漢" pos="名詞-普通名詞-サ変可能" pron="コージ">
# 　　　工事
# 　　</SUW>
# 　　<SUW lemma="請け負い" lForm="ウケオイ" wType="和" pos="名詞-普通名詞-一般" pron="ウケオイ">
# 　　　請負
# 　　</SUW>
# 　　<SUW lemma="金額" lForm="キンガク" wType="漢" pos="名詞-普通名詞-一般" pron="キンガク">
# 　　　金額
# 　　</SUW>
# 　</LUW>
# 　<LUW SL="v" l_lemma="の" l_lForm="ノ" l_wType="和" l_pos="助詞-格助詞" >
# 　　<SUW lemma="の" lForm="ノ" wType="和" pos="助詞-格助詞" pron="ノ">の</SUW>
# 　</LUW>
# 　<LUW B="B" SL="v" l_lemma="動き" l_lForm="ウゴキ" l_wType="和" l_pos="名詞-普通名詞-一般" >
# 　　<SUW lemma="動き" lForm="ウゴキ" wType="和" pos="名詞-普通名詞-一般" pron="ウゴキ">
# 　　　動き
# 　　</SUW>
# 　</LUW>
# （略）
# </sentence>
# ```

# ### コーパスの読み込み
# 

# 多くのコーパスは、*txt*,*csv*,*tsv*, *json*フォーマットで格納されていますので、pythonでこれらのファイルを読み込む方法について紹介します。

# #### ファイル読み込みの基礎
# 
# 読み込み・書き込みいずれの場合も組み込み関数```open()```でファイルを開きます。
# - 第一引数に指定したパス文字列が示すファイルオブジェクトが開かれます。
# - 引数```mode```を```r```とすると読み込みモードでファイルが開かれます。デフォルト値は```r```なので省略してもよい。
# - テキストファイル読み込み時のデコード、書き込み時のエンコードで使われるエンコーディングは引数```encoding```で指定する。日本語環境で使われるものとしては```cp932```, ```euc_jp```, ```shift_jis```, ```utf_8```などがあります。
# 
# 

# In[1]:


with open("./Data/corpus.txt",mode="r",encoding="utf-8") as f:
    s = f.read()
    print(s)


# ```readlines()```メソッドで、ファイル全体を行ごとに分割したリストとして取得できます。

# In[2]:


with open("./Data/corpus.txt",mode="r",encoding="utf-8") as f:
    s = f.readlines()
    for i in s:
        print(i)
        print("-----------")


# #### csvファイルとtsvファイルの読み込み
# 
# csvは、値がカンマで区切られたファイル形式です。
# 
# tsvは、値がタブで区切られたファイル形式です。
# 
# ファイルの読み込みは、```Pandas```を使って行うことがでします。
# 
# ```
# pd.read_csv("example.csv", encoding ="uft-8")
# ```
# 

# #### jsonファイルの読み込み
# 
# JSONは、構造化されたデータをテキスト形式で表現するためのフォーマットです。
# 
# ```
# [
#     {
#         "id": 1,
#         "name": "Taro Yamada",
#         "age": 25,
#         "city": "Tokyo"
#     },
#     {
#         "id": 2,
#         "name": "Hanako Tanaka",
#         "age": 30,
#         "city": "Osaka"
#     },
#     {
#         "id": 3,
#         "name": "Jiro Suzuki",
#         "age": 35,
#         "city": "Kyoto"
#     }
# ]
# ```
# 
# ```Pandas```の```read_json```メソッドを使って読み込めます。
# 
# 特に、改行で区切られたJSONファイルはjsonlで呼ばれます。
# 
# ```
# {"id": 1, "name": "Taro Yamada", "age": 25, "city": "Tokyo"}
# {"id": 2, "name": "Hanako Tanaka", "age": 30, "city": "Osaka"}
# {"id": 3, "name": "Jiro Suzuki", "age": 35, "city": "Kyoto"}
# ```
# 
# jsonl形式のファイルを読み込むには、```read_json```メソッドの```lines```引数に```lines=True```を設定する必要があります。

# ## テキストの前処理
# 
# ### 正規表現によるノイズの除去
# 
# テキストから不要な情報やノイズを取り除き、解析や処理を行いやすくするための前処理が不可欠になります。
# 
# テキストクリーニングを行う際に、正規表現を利用すると非常に柔軟にノイズの除去が可能です。以下、いくつかの一般的なケースを例に、テキストからノイズを除去する方法を紹介します。

# In[3]:


import re

#ウェブページやSNSのテキストからURLを取り除く場合:
text = "この記事はhttp://example.com に投稿されました。"
cleaned_text = re.sub(r'http\S+', '', text)
print(cleaned_text)


# In[4]:


# HTMLタグの除去: HTMLコンテンツからタグを取り除く場合:
text = "今日は2023年9月16日です。"
cleaned_text = re.sub(r'\d', '', text)
print(cleaned_text)


# In[5]:


#テキストからアルファベットのみを取り除く場合:
text = "これはテストtextです。"
cleaned_text = re.sub(r'[a-zA-Z]', '', text)
print(cleaned_text)


# ### トークン化
# 
# コンピュータは、入力として生の文字列を受け取ることができません。その代わりに、テキストがトークン化され、数値ベクトルとしてエンコードされていることが想定しています。
# 
# トークン化は、文字列をモデルで使用される最小単位に分解するステップです。
# 
# 日本語は英語などとは異なり、単語の間に（スペースなどの）区切りをつけないので、文から単語を取り出すこと自体が一つのタスクとなります。文から単語（正確には形態素と呼ばれる意味の最小単位）を取り出す解析を形態素解析といいます。
# 
# 
# イメージとしては以下のように分割します。
# 
# - 分かち書き（文章を形態素で分ける）
# - 品詞わけ（名詞や動詞などに分類する）
# - 原型付与（単語の基本形をだす） 例：食べた ⇒ 食べる、た
# 
# ![](./Figure/token.png)
# 
# 
# 日本語の形態素解析ツールとしては、MeCabやJUMAN++、Janomeなどが挙げられます。ここでは、MeCabを使って形態素解析をしてみましょう。
# 
# 初めてMeCabを使う場合、```!```が付いたコードをJupyter Notebookで実行するか、```!```を除去したコードをターミナルで実行し、MeCabを導入してください。
# 
# ```
# !pip install mecab-python3 # mecab-python3のインストール
# !pip install unidic
# !python -m unidic download # 辞書のダウンロード
# ```

# 結果を確認すると、単語に分割しただけでなく、品詞などの情報も得られました。
# 
# ただ、デフォルトの設定では新語に対する解析は強くないことがわかります。

# In[6]:


# !pip install mecab-python3 # mecab-python3のインストール
# !pip install unidic
# !python -m unidic download # 辞書のダウンロード
import MeCab
import unidic
tagger = MeCab.Tagger() 
print(tagger.parse("友たちと国立新美術館に行った。"))


# この問題は形態素解析器に辞書を追加することである程度解決することが出来ます。特に、*NEologd*という辞書には、通常の辞書と比べて多くの新語が含まれています。
# 
# *NEologd*のインストールについては、以下のベージに参照してください。
# - [Windows](https://resanaplaza.com/2022/05/08/%E3%80%90%E8%B6%85%E7%B5%B6%E7%B0%A1%E5%8D%98%E3%80%91windows-%E3%81%AEpython%EF%BC%8Bmecab%E3%81%A7%E3%83%A6%E3%83%BC%E3%82%B6%E3%83%BC%E8%BE%9E%E6%9B%B8%E3%81%ABneologd%E3%82%92%E4%BD%BF%E3%81%86/)
# - [Mac](https://qiita.com/berry-clione/items/b3a537962c84244a2a09)
# 
# 辞書を指定して、タガーを生成します。

# In[7]:


sample_txt = "友たちと国立新美術館に行った。"
path = "-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd"
m = MeCab.Tagger(path)
print("Mecab ipadic NEologd:\n",m.parse(sample_txt))


# In[ ]:





# ```{note}
# *NEologd*のインストールについては、以下のベージに参照してください。
# - [Windows](https://resanaplaza.com/2022/05/08/%E3%80%90%E8%B6%85%E7%B5%B6%E7%B0%A1%E5%8D%98%E3%80%91windows-%E3%81%AEpython%EF%BC%8Bmecab%E3%81%A7%E3%83%A6%E3%83%BC%E3%82%B6%E3%83%BC%E8%BE%9E%E6%9B%B8%E3%81%ABneologd%E3%82%92%E4%BD%BF%E3%81%86/)
# - [Mac]()
# ```

# 上の例から分かるように、NEologdは「国立新美術館」のような固有表現にも対応し、ソーシャルメディアのテキストなど新語が多数含まれている場合こちらの方が適切です。

# parseToNode()メソッドは、形態素（ノード）ごとに出力をしていきます。

# In[8]:


n=m.parseToNode(sample_txt)


# 最初は、BOS/EOS(文頭／文末)を示す符号となります。

# In[9]:


n.feature


# ```.next```で次のノードに進みます。

# In[10]:


n = n.next


# In[11]:


n.feature


# ところでfeatureは形態素の特徴を表す要素を格納しています。
#   - $0$番目が品詞、$6$番目が基本形であり、今回はこの2つを用います。

# In[12]:


print("品詞: "+n.feature.split(',')[0])
print("基本形: "+n.feature.split(',')[6])


# 文章を解析するためには、whileループを使います。その際、品詞を指定して特定の品詞の基本形だけを取り出すなどという操作が可能です。

# In[13]:


n=m.parseToNode(sample_txt)
while n:
  if n.feature.split(',')[0] in ["名詞","形容詞","動詞"]:
    print(n.feature.split(',')[6])
  n = n.next


# ### ストップワードの除去

# ストップワードとは、助詞など単独で用いられなかったり、一般的に使用されすぎていて文の意味の分析に寄与しない、あるいや逆に使用頻度が少ないため、含めると分析結果が歪んでしまうなどの理由で分析からあらかじめ除外しておく語のことをいいます。
# 
# 一般的には、あらかじめストップワードを辞書に定義しておき、辞書内に含まれる単語をテキスとから除外します。
# 
# 一般的な用語のストップリストは例えばSlothLibプロジェクトから取得することができます。

# In[14]:


import urllib
slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
slothlib_file = urllib.request.urlopen(slothlib_path)
slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']


# ### 単語のID化
# 
# ```{margin} 
# 先取りですが、ここからは機械学習・深層学習の適用するための前処理について解説します。
# ```
# 
# ほとんどの機械学習・深層学習モデルは数値データを入力として受け取りますので、テキストデータを数値形式に変換する必要があります。
# 
# 単語のID化とは、テキストデータを機械学習モデルなどで処理する際に、単語を一意の整数値（ID）に変換するプロセスを指します。これは、テキストデータをベクトルや行列の形でモデルに入力するための前処理として行われます。

# ### パディング
# 
# パディング(Padding)とは、データの一定の長さや形状を持つように調整するためのテクニックです。
# 
# 多くの機械学習・深層学習モデルでは、固定サイズの入力を要求しています。しかし、異なる文書や文はさまざまな長さを持っていますので、テキストデータは可変長である場合が多いです。したがって、すべての文書や文を同じ長さにするためにパディングを使用します。主な手順としては、
# 
# - 最大長を決定
# - 最大長に達するまで、各文の末尾または先頭に特定の「パッドトークン」を追加します。パディングに使用するトークンは、モデルが誤って意味を学習しないように、他の実際の単語やトークンとは異なるものである必要があります。
# - 長すぎる文は、最大長に合わせるために切り捨て
# 
# ![](./Figure/attention_id.png)
# 

# 
