#!/usr/bin/env python
# coding: utf-8

# # BERTによるセンチメント分析
# 
# ## 転移学習とファインチューニング
# 
# 転移学習は、あるタスクの学習で得られた知識を、他の関連するタスクの学習に適用する手法を指します。一般的には、以下のステップで行われることが多いです：
# 
# - 事前学習: 事前学習モデル（pre-trained models)とは、大規模なデータセットを用いて訓練した学習済みモデルのことです。一般的に、大量のデータ（例えば、インターネット上のテキストデータ）を使用して、モデルを事前に学習します。この時点でのモデルは、言語の汎用的な特徴や構造を捉えることができます。
# 
# - ファインチューニング(fine-tuning): 事前学習モデルを、特定のタスクのデータ（例えば、感情分析や質問応答）でファインチューニングします。事前学習モデルでは汎用的な特徴をあらかじめ学習しておきますので、手元にある学習データが小規模でも高精度な認識性能を達成することが知られています。 
# 
# ![](./Figure/fine-tuning_methods.png)

# ## センチメント分析の実装

# In[1]:


get_ipython().system('nvidia-smi')


# ### データセット

# #### Hugging Faceからサンプルデータの取得
# 
# Hugging Faceのには色々なデータセットが用意されております。ここでは、多言語のセンチメントデータセットを例として使用することにします。その中に、英語と日本語のサプセットが含まれます。

# In[2]:


from datasets import load_dataset
#dataset = load_dataset("tyqiangz/multilingual-sentiments", "japanese")
dataset = load_dataset("tyqiangz/multilingual-sentiments", "english")


# #### サンプルデータの確認
# 
# 取得したデータセットの中身を確認します。
# 
# データセットはこのようにtrain, validation, testに分かれています。
# ['text', 'source', 'label']といった情報を持っています。
# 

# In[3]:


dataset


# In[4]:


dataset.set_format(type="pandas")
train_df = dataset["train"][:]
train_df.head(5)


# In[5]:


dataset["train"].features


# In[6]:


import matplotlib.pyplot as plt
train_df["label"].value_counts(ascending=True).plot(kind="barh", title="Train Dataset")


# #### テキストの確認
# 
# Transformerモデルは、最大コンテキストサイズ(maximum context size)と呼ばれる最大入力系列長があります。
# 
# モデルのコンテキストサイズより長いテキストは切り捨てる必要があり、切り捨てたテキストに重要な情報が含まれている場合、性能の低下につながることがあります。

# In[7]:


train_df["text_length"]=train_df["text"].str.len()


# In[8]:


train_df.boxplot(column="text_length", by="label", figsize=(12, 6))


# ### トークン化
# 
# コンピュータは、入力として生の文字列を受け取ることができません。その代わりに、テキストがトークン化され、数値ベクトルとしてエンコードされていることが想定しています。
# 
# トークン化は、文字列をモデルで使用される最小単位に分解するステップです。
# 
# Transformerライブラリー は便利なAutoTokenizerクラスを提供しており、事前学習済みモデルに関連つけられたトークナイザーを素早く使用することができます。

# #### トークナイザの動作確認
# 

# In[9]:


from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


# In[10]:


train_df["text"][0]


# In[11]:


sample_text_encoded = tokenizer(train_df["text"][0])
sample_text_encoded


# 結果にinput_idsとattention_maskが含まれます。
# 
# - input_ids: 数字にエンコードされたトークン
# - attention_mask: モデルで有効なトークンかどうかを判別するためのマスクです。無効なトークン（例えば、PADなど）に対しては、attention_maskを
# として処理します。
# 
# 各batchにおいて、入力系列はbatch内最大系列長までpaddingされます。
# 
#  ![](./Figure/attention_id.png)
# 

# トークナイザの結果は数字にエンコードされているため、トークン文字列を得るには、convert_ids_to_tokensを用います。
# 
# 文の開始が[CLS]、文の終了が[SEP]という特殊なトークンとなっています。

# In[12]:


tokens = tokenizer.convert_ids_to_tokens(sample_text_encoded.input_ids)
print(tokens)


# #### データセット全体のトークン化
# 
# 

# In[13]:


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# In[14]:


dataset.reset_format()


# In[15]:


dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)


# In[16]:


import pandas as pd
sample_encoded = dataset_encoded["train"][0]
pd.DataFrame(
    [sample_encoded["input_ids"]
     , sample_encoded["attention_mask"]
     , tokenizer.convert_ids_to_tokens(sample_encoded["input_ids"])],
    ['input_ids', 'attention_mask', "tokens"]
).T


# ### 分類器の実装
# #### 事前学習モデルの導入
# 
# Transformerライブラリは事前学習モデルの使用ため```AutoModel```クラスを提供します。
# 
# ```AutoModel```クラスはトークンエンコーディングを埋め込みに変換し、エンコーダスタックを経由して**最後の**隠れ状態を返します。
# 

# In[17]:


import torch
from transformers import AutoModel

# GPUある場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)


# 最初に、文字列をエンコーダしてトークンをPyTorchのテンソルに変換する必要があります。
# 
# 結果として得られるテンソルは```[batch_size,n_tokens]```という形状です。

# In[18]:


text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")


# 得られるテンソルをモデルの入力として渡します。
# 
# - モデルと同じデバイス(GPU or CPU)に設置します。
# - 計算のメモリを減らせるため、```torch.no_grad()```で、勾配の自動計算を無効します。
# - 出力には隠れ状態、損失、アテンションのオブジェクトが含まれます。

# In[19]:


inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)


# 隠れた状態テンソルを見ると、その形状は [batch_size, n_tokens, hidden_dim] であることがわかります。つまり、6つの入力トークンのそれぞれに対して、768次元のベクトルが返されます。

# In[20]:


outputs.last_hidden_state.size()


# 分類タスクでは、```[CLS]``` トークンに関連する隠れた状態を入力特徴として使用するのが一般的な方法です。このトークンは各シーケンスの始まりに現れるため、次のように outputs.last_hidden_state に単純にインデックスを付けることで抽出できます。

# In[21]:


outputs.last_hidden_state[:,0].size()


# 最後の隠れ状態を取得する方法がわかりましたので、データ全体に対して処理を行うため、これまでのステップを関数でまとめます。
# 
# そして、データ全体に適用し、すべてのテキストの隠れ状態を抽出します。

# In[22]:


def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


# In[23]:


dataset_encoded.set_format(type="torch", columns=["input_ids", "attention_mask","label"])


# In[24]:


dataset_hidden=dataset_encoded.map(extract_hidden_states, batched=True, batch_size=16)


# #### 分類器の学習
# 
# 前処理されたデータセットには、分類器を学習させるために必要な情報がすべて含まれています。
# 
# 具体的には、隠れ状態を入力特徴量として、ラベルをターゲットとして使用すると、様々な分類アルゴリズムに適用できるだろう。
# 
# ここで、ロジスティック回帰モデルを学習します。

# In[48]:


import numpy as np

X_train = np.array(dataset_hidden["train"]["hidden_state"])
X_valid = np.array(dataset_hidden["validation"]["hidden_state"])
y_train = np.array(dataset_hidden["train"]["label"])
y_valid = np.array(dataset_hidden["validation"]["label"])
X_train.shape, X_valid.shape


# In[49]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)


# In[50]:


lr_clf.score(X_valid, y_valid)


# In[51]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
    
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, ["positive","neutral","negative"])


# #### AutoModelForSequenceClassificationのファインチューニング
# 
# 
# transformerライブラリは、ファインチューニングのタスクに応じてAPIを提供しています。
# 
# 分類タスクの場合、```AutoModel```の代わりに```AutoModelForSequenceClassification```を使用します。
# 
# ```AutoModelForSequenceClassification```が事前学習済みモデルの出力の上に分類器ヘッドを持っており、モデルの設定がより簡単になります。

# In[52]:


from transformers import AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 3

model = (AutoModelForSequenceClassification
    .from_pretrained(model_ckpt, num_labels=num_labels)
    .to(device))


# In[54]:


model


# In[53]:


inputs = tokenizer("普段使いとバイクに乗るときのブーツ兼用として購入しました", return_tensors="pt") # pytorch tensorに変換するためにreturn_tensors="pt"を指定
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)


# #### 学習の準備
# 
# 学習時に性能指標を与える必要があるため、それを関数化して定義しておきます。
# 
# 
# 

# In[55]:


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# 学習を効率化するために、transformerライブラリの```Trainer``` APIを使用します。
# 
# ```Trainer```クラスを初期化する際には、```TrainingArguments```という訓練に関する様々な設定値の集合を引数に与えることで、訓練の設定に関する細かい調整が可能です。

# In[56]:



from transformers import TrainingArguments

batch_size = 16
logging_steps = len(dataset_encoded["train"]) // batch_size
model_name = "sample-text-classification-bert"

training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error"
)


# Trainerクラスで実行します。
# 
# 結果を確認すると、特徴ベースのアプローチよりも精度が改善されることがわかります。

# In[57]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["validation"],
    tokenizer=tokenizer
)
trainer.train()


# ### 学習済みモデルの使用
# 
# #### モデル精度の検証
# 
# 学習済みのモデルを他のデータセットに適用します。
# 
# 

# In[58]:


preds_output = trainer.predict(dataset_encoded["test"])


# In[60]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_preds = np.argmax(preds_output.predictions, axis=1)
y_valid = np.array(dataset_encoded["test"]["label"])
labels = dataset_encoded["train"].features["label"].names

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

plot_confusion_matrix(y_preds, y_valid, labels)


# #### モデル保存
# 
# 

# In[61]:


id2label = {}
for i in range(dataset["train"].features["label"].num_classes):
    id2label[i] = dataset["train"].features["label"].int2str(i)

label2id = {}
for i in range(dataset["train"].features["label"].num_classes):
    label2id[dataset["train"].features["label"].int2str(i)] = i

trainer.model.config.id2label = id2label
trainer.model.config.label2id = label2id


# In[62]:


trainer.save_model(f"./Data/sample-text-classification-bert")


# #### 学習済みモデルの読み込み
# 

# In[63]:


new_tokenizer = AutoTokenizer    .from_pretrained(f"./Data/sample-text-classification-bert")

new_model = (AutoModelForSequenceClassification
    .from_pretrained(f"./Data/sample-text-classification-bert")
    .to(device))


# サンプルテキストで推論の結果を確認します。
# 
# 

# In[64]:


def id2label(x):
    label_dict={0:"positive",1:"neutral",2:"negative"}
    return label_dict[x]


# In[65]:


text1="this week is not going as i had hoped"
text2="awe i love you too!!!! 1 am here i miss you"


# In[66]:



inputs = new_tokenizer(text1, return_tensors="pt")

new_model.eval()

with torch.no_grad():
    outputs = new_model(
        inputs["input_ids"].to(device), 
        inputs["attention_mask"].to(device),
    )
outputs.logits

y_preds = np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=1)
y_preds = [id2label(x) for x in y_preds]
y_preds


# In[67]:


inputs = new_tokenizer(text2, return_tensors="pt")

new_model.eval()

with torch.no_grad():
    outputs = new_model(
        inputs["input_ids"].to(device), 
        inputs["attention_mask"].to(device),
    )
outputs.logits

y_preds = np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=1)
y_preds = [id2label(x) for x in y_preds]
y_preds


# 
