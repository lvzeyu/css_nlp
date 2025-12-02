---
# try also 'default' to start simple
theme: purplin
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: Welcome to Slidev
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply UnoCSS classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# duration of the presentation
duration: 35min
---

# 今までのまとめとこれからの内容


<div @click="$slidev.nav.next" class="mt-12 py-1" hover:bg="white op-10">
  Press Space for next page <carbon:arrow-right />
</div>


<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# 今までのまとめ

## 自然言語処理の核心問題

- どのようにパソコンにテキストを理解させるか？
    - (非構造的な)テキストを構造化された数値形式(ベクトル)に変換
- 単純にあるテキストにどのような単語が含まれているかを数値化するだけでなく、文脈や意味情報も考慮したベクトル化が必要
    - **Word2Vec**では、単語の意味的な類似性を捉えたベクトル表現を学習
- ベクトル化されたテキストに基づく、様々なNLPタスクを解決




---
transition: fade-out
---

# 今までのまとめ


## 深層学習

- 多層ニューラルネットワーク（MLP）を用いて、入力と出力の複雑な関係をモデル化
- テキストをベクトル化し、入力と出力として扱うと、様々なNLPタスクとして定式化できる

<img src="/deep_learning_based_nlp.webp" class="mx-auto" style="max-width: 80%; height: auto;" />

---
transition: fade-out
---

# これからの内容


## 残念ですが、基礎的な深層学習（MLP）では限界がある

- MLPは汎用的モデルであるが、言語特有の構造や特徴について十分に考慮できない部分も多い
    - 語順
    - 文法構造

## より高度なモデルが必要

- RNN: 時間的な依存関係を捉えるモデル
    - 結局色々問題があって、主流の手法にはならなかった
- Transformer: 長距離依存関係を効果的に捉え、並列計算も可能 
    - 現在のNLPの主流モデル（大規模言語モデルも含む）が利用するアーキテクチャ

