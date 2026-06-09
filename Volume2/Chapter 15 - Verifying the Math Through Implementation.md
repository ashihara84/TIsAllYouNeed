# 第15章 実装で確認する数学

**この章のゴール**

ここまで学んだ数学をPyTorchのテンソル計算として実行し、shapeを追いながらAttentionの中心式までを一度確かめられるようになること。部品としてモジュール化したり、Transformerブロックを組んだりするのは次の巻（5・7巻）の仕事です。

## 15.1 この章の目的

ここまで、Transformerを理解するために必要な数学（スカラー・ベクトル・行列・テンソル・shape・内積・行列積・転置・線形変換・softmax・確率分布・cross entropy・微分・勾配・勾配降下法・LayerNorm・Attentionの数式）を学んできました。この章では、それらをPyTorchで実際に確認します。目的は、数学を「読める」だけでなく「コードとして動かせる」ようにすることです。

たとえば `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` を見て「QK^T は何をしているのか、softmax はどの次元にかけるのか、V を掛けるとshapeはどうなるのか」を理解するには、実際にテンソルを作ってshapeを確認するのが効果的です。

この章では「ベクトルを作る → 行列積を計算する → softmaxを実装する → cross entropyを手計算してPyTorchと比べる → 勾配降下法を実装する → Q/K/Vを作ってAttentionを計算する → shapeを確認する」という順で確認します。難しいモデルをいきなり作るのではなく、小さなテンソルで1つずつ意味を確認します。この章のコードはすべて学習用の小さな例で、実用的な性能を出すためのものではありません。

---

## 15.2 ベクトルをPythonで作る

ベクトルは数を一列に並べたもので、PyTorchでは `torch.tensor` で作れます。基本操作（足し算、スカラー倍、長さ、距離）を確認します。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape)   # torch.Size([3])

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)              # tensor([5., 7., 9.])  対応する要素同士を足す
print(2.0 * a)            # tensor([2., 4., 6.])  各要素を2倍

print(torch.norm(torch.tensor([3.0, 4.0])))          # tensor(5.)  sqrt(3^2+4^2)
print(torch.norm(torch.tensor([4.0, 6.0]) - torch.tensor([1.0, 2.0])))  # tensor(5.)  距離
```

```text
ベクトル作成: torch.tensor([...])
足し算: a + b / スカラー倍: 2.0 * a
長さ: torch.norm(a) / 距離: torch.norm(b - a)
```

Transformerでは、トークンは最終的にベクトルとして扱われるので、これらの操作はすべて土台になります。

---

## 15.3 行列をPythonで作る

行列は数を縦横に並べたもので、複数のトークンのベクトルをまとめると行列になります。

```python
import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
print(x.shape)   # torch.Size([2, 3])  2行3列
```

3個のトークンがそれぞれ4次元ベクトルなら shape は `[3, 4]`（`seq_len = 3`, `d_model = 4`）です。実際のTransformerでは複数の文をまとめて処理するので、バッチ付きのテンソルを使います。

```python
import torch

batch_size, seq_len, d_model = 2, 3, 4
x = torch.randn(batch_size, seq_len, d_model)
print(x.shape)   # torch.Size([2, 3, 4])
```

これは「2個の文 / 各文は3トークン / 各トークンは4次元ベクトル」という意味です。Transformerの実装では、このshapeを非常によく使います。

```text
[batch_size, seq_len, d_model]
batch_size: まとめて処理する文の数 / seq_len: トークン数 / d_model: 各トークンのベクトル次元
```

---

## 15.4 行列積をPythonで計算する

行列積はTransformerで非常に重要で、特にAttentionでは `QK^T` が出てきます。行列積のルールは `[a, b] @ [b, c] → [a, c]` です。

```python
import torch

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])           # [2, 2]
B = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])  # [2, 3]
C = A @ B
print(C.shape)   # [2, 3]
```

Transformerでよく出る `QK^T` のshapeを見ます。1つの文なら `[seq_len, d_k] @ [d_k, seq_len] → [seq_len, seq_len]` で、この `[seq_len, seq_len]` がトークン同士の相性スコアです。バッチ付きでは `.T` ではなく `transpose(-2, -1)`（最後の2次元だけ入れ替え）を使います。

```python
import torch

batch_size, seq_len, d_k = 2, 4, 3
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)

scores = Q @ K.transpose(-2, -1)
print(K.transpose(-2, -1).shape, scores.shape)   # [2,3,4] [2,4,4]
```

Attentionの実装では、この形が非常によく出てきます。

---

## 15.5 softmaxを自分で実装する

softmaxは、スコアを合計1の重みに変換する関数です。式は `softmax(x_i) = exp(x_i) / Σ exp(x_j)` で、自分で書けますが、大きな値に弱いので最大値を引く版（数値安定版）にします。

```python
import torch

def stable_softmax(x):
    x = x - x.max()
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum()

scores = torch.tensor([1000.0, 1001.0, 1002.0])
print(stable_softmax(scores))                 # tensor([0.0900, 0.2447, 0.6652])
print(torch.softmax(scores, dim=0))           # 同じ結果
```

Attentionでは各Queryごとに全Keyへの重みを作るので、行列の最後の次元にsoftmaxをかけます。

```python
import torch

scores = torch.tensor([
    [2.0, 1.0, 0.0],
    [0.5, 1.5, 0.0],
    [1.0, 1.0, 2.0],
])
weights = torch.softmax(scores, dim=-1)
print(weights.sum(dim=-1))   # tensor([1., 1., 1.])  各行の合計が1
```

`scores: [batch_size, seq_len, seq_len]` の最後の次元に沿ってsoftmaxをかけることで、各Queryが全Keyを見る重みを作ります。

---

## 15.6 cross entropyを自分で計算する

> ★ちょうつがい注記：ここは cross entropy を **一度実装して確かめる** 段階である。部品として作り込まない（作るのは 5 巻）。

cross entropyは、正解トークンにどれだけ高い確率を割り当てたかを見る損失で、基本は `loss = -log(正解トークンの確率)` です。手計算（softmax → 正解確率を取り出す → -log）とPyTorchの `F.cross_entropy` が一致することを確認します。

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([0.1, 0.2, -0.5, 2.0, 0.0])
target = torch.tensor(3)   # 語彙サイズ5、正解トークンID 3

probs = torch.softmax(logits, dim=-1)
manual_loss = -torch.log(probs[target])                              # tensor(0.5163)
loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))    # tensor(0.5163)
print(manual_loss, loss)
```

重要なのは、`F.cross_entropy` にはsoftmax後の確率ではなくsoftmax前のlogitsを渡すことです（`F.cross_entropy(logits, targets)` が正しい）。

言語モデルでは、logitsは `[batch_size, seq_len, vocab_size]`、targetsは `[batch_size, seq_len]` です。`F.cross_entropy` に渡すために、batchとseq_lenをまとめます。

```python
import torch
import torch.nn.functional as F

batch_size, seq_len, vocab_size = 2, 3, 5
logits = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.tensor([[1, 3, 4], [0, 2, 3]])

B, T, V = logits.shape
loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
```

つまり、すべての位置をまとめて分類問題として扱っています。

---

## 15.7 勾配降下法を小さな例で実装する

`loss = (w - 5)^2`（`w = 5` で最小）を `w = 0` から最小化します。

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

for step in range(10):
    loss = (w - 5) ** 2
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()
    print(step, round(float(w), 3), round(float(loss), 3))
```

`w` が5に近づき、lossが小さくなります。流れは「lossを計算 → `backward()` で勾配を計算 → 勾配の逆方向にwを更新 → `w.grad` をリセット → 繰り返す」です。パラメータ更新は計算グラフに含めたくないので `torch.no_grad()` の中で行い、PyTorchでは勾配が蓄積されるので毎回 `w.grad.zero_()` でリセットします。

optimizerを使う形でも書けます。

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(10):
    loss = (w - 5) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` の3行が、PyTorchの学習ループの基本です。

---

## 15.8 小さな線形モデルを学習する

`y = 2x + 1` という関係を `nn.Linear` で学習します。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(1000):
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model.weight.data, model.bias.data)   # weight ≒ 2, bias ≒ 1
```

流れ（入力x → model → 予測 → 正解と比較 → loss → backward → optimizer step）は、ニューラルネットワークの学習そのものです。Transformerでも基本の流れ（token_ids → Transformer → logits → cross entropy loss → backward → optimizer step）は同じで、モデルが大きくなっても学習ループの考え方は変わりません。

---

## 15.9 embeddingと出力層で小さな言語モデル風にする

本物のTransformerはまだ使わず、embeddingと出力層だけで次トークン予測の形を確認します。入力と正解を1つずらし、学習ループまで作ります。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

token_ids = torch.tensor([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
])

inputs = token_ids[:, :-1]    # [2, 4]
targets = token_ids[:, 1:]    # [2, 4]

vocab_size, d_model = 10, 8
embedding = nn.Embedding(vocab_size, d_model)
output_layer = nn.Linear(d_model, vocab_size)

params = list(embedding.parameters()) + list(output_layer.parameters())
optimizer = torch.optim.AdamW(params, lr=0.01)

for step in range(100):
    hidden = embedding(inputs)       # [2, 4, 8]
    logits = output_layer(hidden)    # [2, 4, 10]
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, round(float(loss), 4))
```

shapeの流れは「inputs `[batch, seq]` → embedding → hidden `[batch, seq, d_model]` → 出力層 → logits `[batch, seq, vocab_size]`」です。このコードは小さいですが、言語モデル学習の骨格を含んでいます。本物のTransformerでは embedding と出力層の間に Transformer block が入りますが、loss計算と学習ループは基本的に同じです。

---

## 15.10 Q/K/Vを作ってAttentionを計算する

入力 `x` からQ/K/Vを作り、Attentionの中心部分を計算します。

```python
import torch
import torch.nn as nn
import math

batch_size, seq_len, d_model = 2, 4, 8
x = torch.randn(batch_size, seq_len, d_model)

w_q = nn.Linear(d_model, d_model)
w_k = nn.Linear(d_model, d_model)
w_v = nn.Linear(d_model, d_model)

q, k, v = w_q(x), w_k(x), w_v(x)   # 各 [2, 4, 8]

d_k = q.size(-1)
scores = q @ k.transpose(-2, -1)   # [2, 4, 4]
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
out = weights @ v                  # [2, 4, 8]

print("scores:", scores.shape, "weights:", weights.shape, "out:", out.shape)
```

このコードはSelf-Attentionの中心部分で、数式と次のように対応します。

```text
scores = q @ k.transpose(-2, -1)      = QK^T
scores = scores / sqrt(d_k)           = QK^T / sqrt(d_k)
weights = softmax(scores)             = softmax(QK^T / sqrt(d_k))
out = weights @ v                     = softmax(QK^T / sqrt(d_k))V
```

つまり、この数式をそのままPyTorchに落とすと、ほぼこのコードになります。

---

## 15.11 causal mask付きAttentionを一度計算して確かめる

Decoder-only Transformer（GPT系）では未来のトークンを見てはいけないので、causal maskを使います。maskは下三角行列で、`torch.tril` で作れます。

```python
import torch
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = weights @ v
    return out, weights

batch_size, seq_len, d_k, d_v = 1, 4, 8, 8
q = torch.randn(batch_size, seq_len, d_k)
k = torch.randn(batch_size, seq_len, d_k)
v = torch.randn(batch_size, seq_len, d_v)

mask = torch.tril(torch.ones(seq_len, seq_len))
out, weights = scaled_dot_product_attention(q, k, v, mask)
print(weights.sum(dim=-1))   # 各行の合計は1
```

このとき未来方向の重みは0になります。1番目のトークンは未来を見られないので重みは `[1.0, 0.0, 0.0, 0.0]`、2番目は `[重み, 重み, 0.0, 0.0]` のようになります。このように、maskを使うとAttentionの見える範囲を制御できます。Decoder-only Transformerを実装するときには、このcausal maskが重要です。

---

## 15.12 LayerNormと残差接続を実装で確認する

Transformer blockでは、AttentionだけでなくLayerNormと残差接続も重要です。サブレイヤーの代わりに簡単な線形層を使い、Post-LNとPre-LNの両方を確認します。

```python
import torch
import torch.nn as nn

batch_size, seq_len, d_model = 2, 4, 8
x = torch.randn(batch_size, seq_len, d_model)

sublayer = nn.Linear(d_model, d_model)
layer_norm = nn.LayerNorm(d_model)

# Post-LN: LayerNorm(x + Sublayer(x))
y_post = layer_norm(x + sublayer(x))

# Pre-LN: x + Sublayer(LayerNorm(x))
y_pre = x + sublayer(layer_norm(x))

print(x.shape, y_post.shape, y_pre.shape)   # 全て [2, 4, 8]
```

どちらもshapeは `[batch_size, seq_len, d_model]` のまま変わりません。Transformer blockを何層も重ねられるのは、入力と出力のshapeが同じだから（`Block: [B, T, C] → [B, T, C]`、B=batch_size, T=seq_len, C=d_model）です。残差接続とLayerNormは、Transformerの安定した学習に重要です。この章ではまず、実装上の形（`layer_norm(x + sublayer(x))` または `x + sublayer(layer_norm(x))`）を押さえます。

---

## 15.13 この章のまとめ

この章では、ここまで学んだ数学をPyTorchで確認しました。ベクトル（作成・足し算・スカラー倍・長さ・距離）、行列とテンソル（`[batch_size, seq_len, d_model]`）、行列積（Attentionの `QK^T = Q @ K.transpose(-2, -1)`）、softmax（スコアを合計1の重みに変換）、cross entropy（`F.cross_entropy(logits, targets)`、logitsを渡す）、勾配降下法（`zero_grad` / `backward` / `step` の3行）、Q/K/VからのAttention計算を、すべて小さな例で動かしました。

Q/K/Vを作ってAttentionを計算したコードは、数式 `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` に対応しています。

```python
q, k, v = w_q(x), w_k(x), w_v(x)
scores = q @ k.transpose(-2, -1)
weights = torch.softmax(scores, dim=-1)
out = weights @ v
```

さらに、causal mask付きAttention（`scores.masked_fill(mask == 0, float("-inf"))`）と、LayerNorm・残差接続を実装で確認しました。これらがどう組み合わさって `LayerNorm(x + Sublayer(x))` の形になるのかは、論文の式として 6 巻で読み、部品として組むのは 5 巻、統合して動かすのは 7 巻の仕事です。この巻では「数式がコードに対応していること」を一度確かめれば十分です。

```text
x + Attention(LayerNorm(x))   ← この「形」を読めるようになれば十分
x + FeedForward(LayerNorm(x)) ← 部品化・統合は次の巻へ
```

この章で特に重要なのは、次の理解です。

```text
数学の式はPyTorchのテンソル計算に対応している
Transformer実装ではshapeを追うことが非常に重要である
Attentionの中心式は数行のPyTorchコードで書ける
loss.backward()によって勾配が計算される
optimizer.step()によってパラメータが更新される
```

### 実行用サンプル

この章の内容は、`examples/` のコードでも確認できます。

```bash
python3 examples/01_softmax.py
python3 examples/02_cross_entropy.py
python3 examples/03_attention.py
```

本文を読んだあとにコードを実行すると、shapeと計算結果を対応づけやすくなります。

### 確認問題

次の2行は、それぞれAttentionの式のどの部分に対応しているでしょうか。

```python
weights = torch.softmax(scores, dim=-1)
out = weights @ v
```

答えは次の通りです。

```text
weights = softmax(QK^T / sqrt(d_k))
out = weights @ V
```

次章では、この数学編の最後として、Transformer実装に進む前の確認を行います。これまで学んだ内容をチェックリストとして整理し、次に「ニューラルネットワークの基本」または「Transformer実装」に進むための準備を確認します。
