# 第6章 線形変換

**この章のゴール**

`Wx + b` を「ベクトルを別の表現へ変換する操作」として理解し、embeddingからQ/K/Vを作る流れを読めるようになること。

## 6.1 線形変換とは何か

**線形変換**とは、ざっくり言えば、ベクトルに行列を掛けて別のベクトルに変換することです。

たとえば `x = [1.0, 2.0]` に次の行列を掛けます。

```text
W = [
  [1.0, 0.0],
  [0.0, 2.0]
]

xW = [1.0, 4.0]
```

2番目の成分が2倍になりました。つまりこの `W` は `[x1, x2] → [x1, 2*x2]` という働きを持ちます。これが線形変換の基本イメージです。

```text
入力ベクトル → 行列を掛ける → 出力ベクトル
```

ニューラルネットワークでは、入力を少しずつ別の表現へ変換していきます。Transformerも同じで、入力トークンのベクトルはそのまま使われず、何度も別のベクトルに変換されます。たとえばSelf-Attentionでは、入力ベクトルからQuery・Key・Valueを作ります。この変換に使われるのが線形変換です。

---

## 6.2 行列はベクトルを別のベクトルに変換する

行列を掛けると、ベクトルの値が変わるだけでなく、次元数も変えられます。

たとえば3次元ベクトル `x = [1.0, 2.0, 3.0]`（shape `[3]`）に、shape `[3, 2]` の行列を掛けると、2次元ベクトルになります。

```text
[3] @ [3, 2] → [2]
```

計算してみます。

```text
x @ W
= [1.0, 2.0, 3.0] @ [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

成分1: 1.0*1.0 + 2.0*0.0 + 3.0*1.0 = 4.0
成分2: 1.0*0.0 + 2.0*1.0 + 3.0*1.0 = 5.0

x @ W = [4.0, 5.0]
```

このように行列を使うと次元数を変えられます。逆に `[2] @ [2, 4] → [4]` のように増やすこともできます。

Transformerでは、この次元変換が多く出てきます。入力の次元を `d_model`、Queryの次元を `d_k` とすると、`W_Q` のshapeは `[d_model, d_k]` で、次の流れになります。

```text
入力ベクトル x ([d_model])
↓ W_Q で変換
Queryベクトル q ([d_k])
```

---

## 6.3 `Wx + b` の意味

ニューラルネットワークの説明では、よく次の式が出てきます。

```text
y = Wx + b
```

書き方の流儀によっては `y = xW + b` とも書かれます。この教科書では、Transformer実装でshapeを追いやすいように、基本的に `y = xW + b` の形で考えます。意味は次の通りです。

```text
x: 入力ベクトル / W: 重み行列 / b: バイアスベクトル / y: 出力ベクトル
```

たとえば `x = [1.0, 2.0, 3.0]`（3次元）を2次元に変換するなら、`W` は `[3, 2]` です。`xW = [4.0, 5.0]` が得られたとして、ここにバイアス `b = [0.5, -0.5]` を足します。

```text
y = xW + b
  = [4.0, 5.0] + [0.5, -0.5]
  = [4.5, 4.5]
```

つまり `xW` でベクトルを変換し、最後に `b` で値を少しずらしています。`W` と `b` は学習されるパラメータで、最初はランダムに近い値から始まり、損失が小さくなるように勾配で更新されます。

PyTorchでは、これは `nn.Linear` で実装します。

```python
import torch
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0])
linear = nn.Linear(3, 2)
y = linear(x)
print(y.shape)   # torch.Size([2])
```

これは「3次元ベクトル → `nn.Linear(3, 2)` → 2次元ベクトル」という変換です。

---

## 6.4 ニューラルネットワークにおける重み行列

重み行列は、入力をどう変換するかを決めるパラメータです。

```python
import torch.nn as nn

linear = nn.Linear(3, 2)
print(linear.weight.shape)   # torch.Size([2, 3])
print(linear.bias.shape)     # torch.Size([2])
```

ここで注意です。PyTorchの `nn.Linear(in_features, out_features)` では、重みのshapeは `[out_features, in_features]` になります。つまり `nn.Linear(3, 2)` の重みは `[2, 3]` です。

数学的な説明では `xW + b` と書いて `W` を `[3, 2]` と考えることがありますが、PyTorch内部では重みが `[2, 3]` で持たれ、その転置を使う形で計算されます。最初はこの違いに深入りしなくて大丈夫です。実装上、重要なのは次のことだけです。

```text
nn.Linear(in_features, out_features)
= 最後の次元を in_features から out_features に変換する
```

たとえば複数のベクトルをまとめて変換しても、先頭の次元はそのまま残ります。

```python
import torch
import torch.nn as nn

x = torch.randn(5, 3)
linear = nn.Linear(3, 2)
y = linear(x)
print(x.shape, "->", y.shape)   # [5, 3] -> [5, 2]
```

Transformerでも同じです。入力が `[batch_size, seq_len, d_model]` のとき、`nn.Linear(d_model, d_k)` を適用すると `[batch_size, seq_len, d_k]` になります。つまり `nn.Linear` は各トークンのベクトルに同じ変換を適用していると考えられます。

---

## 6.5 入力ベクトルを別の表現に変換する

線形変換の役割は、入力ベクトルを別の表現に変換することです。ここでいう「別の表現」とは、同じ情報を別の見方で表したものです。

あるトークンのembedding vector `x` があるとき、Self-Attentionではこれをそのまま使わず、Query・Key・Valueという3つの役割に分けます。同じ `x` から、異なる重み行列で3種類のベクトルを作ります。

```text
q = xW_Q + b_Q
k = xW_K + b_K
v = xW_V + b_V
```

同じ `x` を入力しても、使う重み行列が違うので出力も違います。これは人間の言葉でいえば、同じ対象を別の観点から見るようなものです。たとえばある人を「採用候補者として見る」「顧客として見る」「友人として見る」では注目する情報が変わります。同じ元情報でも、目的によって取り出したい表現が変わるのです。

Self-Attentionでも同じように、同じトークンベクトルから目的の違う3種類の表現を作っています。

```text
Query: 自分が探しているもの
Key:   自分が持っているラベル
Value: 実際に渡す中身
```

---

## 6.6 embeddingからQ/K/Vを作る

ここでは、Transformerで非常に重要な、embeddingからQ/K/Vを作る流れを見ます。

入力はトークンIDです（shape `[batch_size, seq_len]`）。これをembedding層に通すと、各トークンがベクトルになります（`[batch_size, seq_len, d_model]`）。この `x` からQuery・Key・Valueを作ります。

```python
import torch
import torch.nn as nn

batch_size, seq_len, vocab_size, d_model = 2, 5, 100, 8

token_ids = torch.tensor([
    [12, 45, 98, 3, 7],
    [8, 21, 21, 56, 4],
])

embedding = nn.Embedding(vocab_size, d_model)
x = embedding(token_ids)

w_q = nn.Linear(d_model, d_model)
w_k = nn.Linear(d_model, d_model)
w_v = nn.Linear(d_model, d_model)

q, k, v = w_q(x), w_k(x), w_v(x)

print("token_ids:", token_ids.shape)  # [2, 5]
print("x:", x.shape)                  # [2, 5, 8]
print("q:", q.shape)                  # [2, 5, 8]
```

shapeの流れは次の通りです。

```text
token_ids: [batch_size, seq_len]
↓ embedding
x:         [batch_size, seq_len, d_model]
↓ Linear
q, k, v:   [batch_size, seq_len, d_model]
```

この例では簡単のためQ/K/Vの次元をすべて `d_model` と同じにしていますが、より一般的には次のように考えます。

```text
w_q: nn.Linear(d_model, d_k)  → q: [batch_size, seq_len, d_k]
w_k: nn.Linear(d_model, d_k)  → k: [batch_size, seq_len, d_k]
w_v: nn.Linear(d_model, d_v)  → v: [batch_size, seq_len, d_v]
```

重要なのは、`nn.Linear` が最後の次元だけを変換することです。この処理によって、各トークンのembedding vectorがAttention用のQuery・Key・Valueに変換されます。

---

## 6.7 Transformerの中の線形層

Transformerでは、線形層が多くの場所で使われます。

```text
Q/K/Vを作る線形層
Multi-Head Attention後の出力線形層
Feed Forward Networkの中の線形層
最終的な語彙への出力層
```

**Q/K/Vを作る線形層**は、入力ベクトルをAttention用の3種類の表現に変換します（`q = W_Q(x)` など）。

**Multi-Head Attention後の出力線形層**は、複数headの出力を結合したあと、もう一度線形層を通して `d_model` 次元に戻します。

**Feed Forward Networkの中の線形層**は、典型的には「Linear → 活性化関数 → Linear」という構造です。元論文では `d_model = 512`、中間次元 `d_ff = 2048` のように、いったん大きな次元に広げてから戻します。

```text
[batch_size, seq_len, d_model]
↓ Linear(d_model, d_ff)
[batch_size, seq_len, d_ff]
↓ 活性化関数
↓ Linear(d_ff, d_model)
[batch_size, seq_len, d_model]
```

**語彙への出力層**は、各位置のベクトルを語彙サイズ分のスコアに変換します。

```text
hidden state: [batch_size, seq_len, d_model]
↓ Linear(d_model, vocab_size)
logits:       [batch_size, seq_len, vocab_size]
```

たとえば語彙が50,000個あれば、各位置について50,000個のスコア（次トークン候補のスコア）を出します。このように線形層は、Transformerの内部で表現を変換する基本部品です。

---

## 6.8 PyTorchで線形変換を確認する

`nn.Linear` は、入力の最後の次元だけを変換し、先頭の次元はそのまま残します。3つのshapeで確認します。

```python
import torch
import torch.nn as nn

linear = nn.Linear(3, 2)

a = torch.tensor([1.0, 2.0, 3.0])   # [3]
b = torch.randn(4, 3)               # [4, 3]
c = torch.randn(2, 5, 3)            # [batch, seq, 3]

print(linear(a).shape)   # [2]
print(linear(b).shape)   # [4, 2]
print(linear(c).shape)   # [2, 5, 2]
```

いずれも最後の次元 `3` が `2` に変わり、それ以外の次元は変わりません。

```text
[3]        → [2]
[4, 3]     → [4, 2]
[2, 5, 3]  → [2, 5, 2]
```

この「最後の次元だけを変換する」という性質は、Transformer実装で非常によく使います。

---

## 6.9 PyTorchでQ/K/Vを作る

ここでは、実際にPyTorchでQ/K/Vを作り、Attentionの中心部分まで一気にshapeを追います。

```python
import torch
import torch.nn as nn
import math

batch_size, seq_len, d_model, d_k, d_v = 2, 5, 8, 4, 6

x = torch.randn(batch_size, seq_len, d_model)

w_q = nn.Linear(d_model, d_k)
w_k = nn.Linear(d_model, d_k)
w_v = nn.Linear(d_model, d_v)

q, k, v = w_q(x), w_k(x), w_v(x)

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
out = weights @ v

print("x:", x.shape)            # [2, 5, 8]
print("q:", q.shape)            # [2, 5, 4]
print("k:", k.shape)            # [2, 5, 4]
print("v:", v.shape)            # [2, 5, 6]
print("scores:", scores.shape)  # [2, 5, 5]
print("weights:", weights.shape)# [2, 5, 5]
print("out:", out.shape)        # [2, 5, 6]
```

shapeの流れを追います。

```text
x:       [2, 5, 8]
q, k:    [2, 5, 4]   (d_model=8 → d_k=4)
v:       [2, 5, 6]   (d_model=8 → d_v=6)

scores = q @ k.transpose(-2,-1):
  [2, 5, 4] @ [2, 4, 5] → [2, 5, 5]

out = weights @ v:
  [2, 5, 5] @ [2, 5, 6] → [2, 5, 6]
```

ここで重要なのは、`q` と `k` の最後の次元が同じ `d_k` であることです。AttentionではQueryとKeyの内積を取るので、次元が一致している必要があります。一方 `v` の次元 `d_v` は、理屈の上では `d_k` と違っていても構いません。Valueは最後に重みに応じて混ぜられる中身だからです。

このコードはSelf-Attentionの中心部分にかなり近いですが、この章の目的はAttentionの完全な理解ではありません。ここで重要なのは、線形層によって入力 `x` から `q, k, v` を作っていることです。

---

## 6.10 線形変換だけでは足りない理由

線形変換は重要な部品ですが、これだけを何層重ねても表現力はあまり増えません。なぜなら、線形変換を何度重ねても、全体としてまた1つの線形変換にまとめられるからです。

```text
h = xW_1
y = hW_2 = xW_1W_2 = xW
```

`W_1W_2` もまた1つの行列なので、結局1回の線形変換と同じ形になります。

そこでニューラルネットワークでは、線形変換の間に **非線形な処理**（活性化関数）を入れます。

```text
Linear → 活性化関数 → Linear
```

活性化関数にはReLU、GELU、tanhなどがあります。TransformerのFeed Forward Networkでも、線形層の間に活性化関数が入ります。この非線形性があることで、ニューラルネットワークは複雑な関数を表現できるようになります。

なお、Self-AttentionでQ/K/Vを作る部分は線形変換が基本で、その後の内積・softmax・重み付き和でトークン同士の情報を混ぜます。つまりTransformer全体では、線形変換・内積・softmax・重み付き和・活性化関数・正規化・残差接続といった操作が組み合わさっています。この章では、その中でも特に基本となる線形変換を扱いました。

---

## 6.11 まとめ

この章では、線形変換について学びました。線形変換とは、ベクトルに行列を掛けて別のベクトルに変換することです。基本式は次の通りです。

```text
y = xW + b
（x: 入力 / W: 重み行列 / b: バイアス / y: 出力）
```

`W` と `b` は学習されるパラメータで、損失が小さくなるように更新されます。PyTorchでは `nn.Linear(in_features, out_features)` で実装し、これは最後の次元を `in_features` から `out_features` に変換します。

```text
[batch_size, seq_len, d_model]
↓ Linear(d_model, d_k)
[batch_size, seq_len, d_k]
```

Transformerでは、線形変換がQ/K/V生成・Multi-Head Attentionの出力層・Feed Forward Network・語彙への出力層など、さまざまな場所で使われます。特にSelf-Attentionでは、入力ベクトルからQuery・Key・Valueを作るために線形変換を使います。

```python
q = w_q(x)
k = w_k(x)
v = w_v(x)
```

この章で特に重要なのは、次の理解です。

```text
行列はベクトルを別のベクトルに変換する
線形層は最後の次元を変換する
TransformerではQ/K/Vを線形層で作る
QとKは内積を取るので同じ次元にする必要がある
線形変換はTransformerの基本部品である
```

次章では、softmaxについて学びます。softmaxは、Attention scoreを「どのトークンをどれくらい見るか」という重みに変換するために使われます。Transformerの中心式では `softmax(QK^T / sqrt(d_k))` の部分です。

---

<!-- chapter-nav:start -->

[← 第5章 行列の基本](Chapter%205%20-%20Matrix%20Basics.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第7章 softmax →](Chapter%207%20-%20Softmax.md)

<!-- chapter-nav:end -->
