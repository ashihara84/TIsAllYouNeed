# 第16章 まとめと次の巻への橋渡し（自己チェックリスト）

**この章のゴール**

数学編で学んだshape、Attention、loss、学習ループを総点検し、次のTransformer実装編へ進める状態か確認すること。

## 16.1 この章の目的

この章では、ここまで学んだ内容を整理します。この教科書の目的は数学そのものを深く極めることではなく、次の式を読めるようになり、PyTorchで実装できるようになることでした。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

この式には、ベクトル・行列・内積・転置・行列積・softmax・shape・線形変換・確率分布・勾配・正規化といった要素が含まれていて、ここまでの章で一通り見てきました。特に重要なのは次の3つです。

```text
shapeを追えること
Attentionの式をコードにできること
学習ループの意味がわかること
```

細かい数式をすべて暗記している必要はありませんが、テンソルのshapeがどう変化するかを追えないと実装はすぐに詰まります。Self-Attentionでは次のshapeが出てきます。

```text
x:       [batch_size, seq_len, d_model]
q, k:    [batch_size, seq_len, d_k]
v:       [batch_size, seq_len, d_v]
scores:  [batch_size, seq_len, seq_len]
weights: [batch_size, seq_len, seq_len]
out:     [batch_size, seq_len, d_v]
```

この流れを読めることが、Transformer実装の土台です。また、言語モデルとして学習させるには「token_ids → inputs, targetsを作る → model(inputs) → logits → cross entropy loss → loss.backward() → optimizer.step()」という流れも必要です。この章では、ここまでの内容を確認し、次に進む準備を整えます。

---

## 16.2 ここまでで理解しておきたいチェックリスト

数学編を終えた時点で、次のことが説明できれば十分です。

- **shape**：`[batch_size, seq_len, d_model]` を見て「文の数 / トークン数 / 各トークンのベクトル次元」と説明できる。
- **embedding**：`token_ids: [batch_size, seq_len]` → embedding → `x: [batch_size, seq_len, d_model]` と説明できる。
- **Q/K/V**：`q = w_q(x)`, `k = w_k(x)`, `v = w_v(x)` のように、xから線形変換で作ると説明できる。
- **Attention score**：`scores = q @ k.transpose(-2, -1)` が「QueryとKeyの内積をまとめて計算している」と説明できる。
- **softmax**：`weights = torch.softmax(scores, dim=-1)` が「各Queryが各Keyを見る重みを作っている」と説明できる。
- **Valueを混ぜる**：`out = weights @ v` が「Attention weightに応じてValueを重み付き和している」と説明できる。
- **causal mask**：Decoder-only Transformerでは未来のトークンを見ないようにmaskすると説明できる。
- **cross entropy**：logitsとtargetsからlossを計算し、正解トークンの確率が高いほどlossは小さいと説明できる。
- **学習ループ**：`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` の意味を説明できる。
- **LayerNormと残差接続**：LayerNormは値のスケールを整える、残差接続は `x + Sublayer(x)`、Transformer blockでは両方が重要、と説明できる。

このチェックリストを完全に暗記する必要はありません。しかし、見たときに意味が追える状態になっていれば、Transformer実装に進めます。

---

## 16.3 次に学ぶべきこと

この数学編の次に学ぶべきことは、**ニューラルネットワークの基本** です。すでに機械学習の基本を学び、この数学編でTransformerに必要な最小限の数学を見ました。次に必要なのは、ニューラルネットワークを部品として理解することです。

特に重要なのは、ニューラルネットワークとは何か、線形層、活性化関数、多層パーセプトロン、forwardとbackward、パラメータ、optimizer、ミニバッチ学習、過学習、正則化、Dropout、LayerNorm、残差接続などです。このうち線形層・LayerNorm・残差接続はこの数学編でもすでに少し出てきました。次の教科書では、それらをニューラルネットワーク全体の文脈で整理するとよいです。

その後、自然言語処理の基本（tokenization、vocab、embedding、language model、next token prediction、seq2seq、RNN、LSTM、Attention）に進み、最後にTransformer本体（Self-Attention、Multi-Head Attention、Positional Encoding、Feed Forward Network、Residual Connection、LayerNorm、Encoder、Decoder、Decoder-only Transformer）に進みます。学習の流れは次のようになります。

```text
機械学習の基本 → 数学の最低限 → ニューラルネットワークの基本
→ 自然言語処理の基本 → Attention以前の流れ → Transformer → 小さなTransformer実装
```

この数学編は、その中の2番目にあたります。ここまで理解できれば、Transformerの数式で使われる数学的な道具はかなり揃っています。

---

## 16.4 まとめ

この章では、Transformer実装に進む前の確認をしました。Transformerで最も重要なshapeは `[batch_size, seq_len, d_model]`（文の数・トークン数・各トークンのベクトル次元）です。

主な流れを確認しました。embeddingは `token_ids: [batch_size, seq_len]` → `x: [batch_size, seq_len, d_model]`、Q/K/Vは線形変換（`q = w_q(x)` など）で作り、Attentionの中心計算は次の通りです。

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
out = weights @ v
```

これは `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` に対応します。causal mask（`scores.masked_fill(mask == 0, float("-inf"))`）はDecoder-only Transformerで未来のトークンを見ないために使います。

言語モデルのloss計算では、logits `[batch_size, seq_len, vocab_size]` と targets `[batch_size, seq_len]` を次のようにまとめます。

```python
B, T, V = logits.shape
loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
```

PyTorchの学習ループの基本は `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` の3行です。次へ進む順番は「機械学習の基本 → 数学の最低限 → ニューラルネットワークの基本 → 自然言語処理の基本 → Attention以前の流れ → Transformer → 小さなTransformer実装」です。

この数学編で学んだ内容は、Transformerの式を読むための道具です。特に重要なのは、次の一文です。

```text
Attentionは、QueryとKeyの内積でトークン同士の相性スコアを作り、softmaxで重みに変換し、その重みに応じてValueを混ぜる仕組みである。
```

この説明ができて、対応するPyTorchコードを読めるなら、この数学編の目的は達成できています。

### 最後の確認問題

次のshapeを順番に埋めてください。

```text
x:       [batch_size, seq_len, d_model]
q:       [batch_size, seq_len, d_model]
k:       [batch_size, seq_len, d_model]
scores:  ?
weights: ?
out:     ?
```

答えは次の通りです。

```text
scores:  [batch_size, seq_len, seq_len]
weights: [batch_size, seq_len, seq_len]
out:     [batch_size, seq_len, d_model]
```

`examples/03_attention.py` を実行すると、Q/K/Vからscores・weights・outまでのshapeの流れとcausal maskの動きを確認できます。

次は、ニューラルネットワークの基本に進むとよいです。
