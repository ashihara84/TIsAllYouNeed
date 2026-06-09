# 第8章 確率分布

**この章のゴール**

softmaxの出力を確率分布として読み、言語モデルが「次のトークン候補への確率分布」を出すことを理解すること。

## 8.1 確率とは何か

この章では **確率分布** について学びます。言語モデルは最終的に「次のトークンが何であるか」の確率分布を出すので、これはとても重要です。

たとえば `I love` という文の次に来る候補に、言語モデルは「どれくらいありそうか」を数値で出します。

```text
dogs : 0.30
cats : 0.20
music: 0.10
you  : 0.40
```

これは「次のトークンが dogs である確率は 0.30」のように読めます。確率には基本的に次の性質があります。

```text
0以上である
1以下である
すべての候補の確率を足すと1になる
```

たとえば `[0.30, 0.20, 0.10, 0.40]` は合計1で確率として自然ですが、`[0.30, -0.20, 0.10, 0.80]` はマイナスがあるため、`[0.30, 0.20, 0.10, 0.10]` は合計が0.70で1にならないため、そのままでは確率分布として扱えません。

Transformerでは、softmaxによってスコア（logits）を確率のような値（probabilities）に変換します。

---

## 8.2 確率分布とは何か

**確率分布** とは、複数の候補に対して確率を割り当てたものです。たとえば公平なサイコロなら、各目が出る確率はすべて `1/6` で、合計すると1になります。6が出やすい偏ったサイコロ `[0.10, 0.10, 0.10, 0.10, 0.10, 0.50]` も、合計1なら確率分布です。

言語モデルでも同じで、サイコロの目の代わりにトークン候補があります。語彙が5個 `["I", "you", "dog", "cat", "."]` のとき、ある文脈の次に来るトークンの確率分布は次のようになるかもしれません。

```text
I: 0.05, you: 0.10, dog: 0.30, cat: 0.25, .: 0.30   （合計 1.00）
```

このように確率分布とは「候補全体に対して、合計1になるように確率を割り当てたもの」です。Transformerを使った言語モデルでは語彙全体に対する確率分布を出します。語彙数が50,000なら、長さ `vocab_size` の配列で、すべての値を足すと1になります。

---

## 8.3 カテゴリ分布

言語モデルでよく出てくる確率分布は **カテゴリ分布** です。これは複数の候補の中から1つを選ぶときの確率分布で、サイコロも次トークン予測もこの例です。

語彙が5個 `["I","you","dog","cat","."]` で、モデルが `[0.05, 0.10, 0.30, 0.25, 0.30]` という確率分布を出したとします。この分布から次のトークンを選ぶ方法には、一番確率の高いものを選ぶ **argmax** と、確率に従ってランダムに選ぶ **sampling** があります。

```text
sampling: dog は30%、cat は25%、you は10%の確率で選ばれる
```

言語モデルの文章生成は、この確率分布から1つ選び、それを文脈に追加し、また次の確率分布を出す、という繰り返しです。

```text
文脈 → 次トークンの確率分布 → トークンを1つ選ぶ → 文脈に追加 → 繰り返し
```

重要なのは、モデルが「文章」を一気に出しているのではなく、次のトークンの確率分布を出して1つ選ぶ処理を繰り返している、という点です。

---

## 8.4 言語モデルは次のトークンの確率分布を出す

言語モデルの基本的な仕事は、次のトークンを予測することです。`I love` という文脈に対して、モデルは候補にスコアを出し、softmaxを通して確率分布に変換します。

```text
dogs: 0.30, cats: 0.20, music: 0.10, you: 0.35, .: 0.05
```

正解が `you` なら、`you` に高い確率（0.35）を割り当てていれば良い予測、低い確率（0.01）なら悪い予測です。学習では正解トークンの確率が高くなるようにパラメータを調整します。そのための代表的な損失関数がcross entropy（次章）です。

この章でまず押さえるのは「言語モデルの出力は、語彙全体に対する確率分布である」ことです。実際のTransformerでは、最終層の出力はまず `logits` というスコアになります。

```text
hidden state → Linear(d_model, vocab_size) → logits
```

この `logits`（例: `[2.0, 0.5, -1.0, 3.0, 0.0]`）はまだ確率ではありません。softmaxに通すと確率分布（例: `[0.242, 0.054, 0.012, 0.657, 0.033]`）になります。

---

## 8.5 語彙全体に対する確率分布

言語モデルでは、次のトークン候補は語彙全体です。語彙とはモデルが扱えるトークンの集合です。たとえば `vocab = ["I","you","love","dogs","cats","."]`（`vocab_size = 6`）のとき、`I love` という文脈に対してモデルは6個の候補すべてにスコアを出します。

```text
logits = [0.1, 2.0, -1.0, 1.5, 1.2, 0.0]
```

この時点ではマイナスもあり合計も1ではないので、softmaxをかけます。

```text
probabilities = softmax(logits)
= [I: 0.071, you: 0.474, love: 0.024, dogs: 0.287, cats: 0.213, .: 0.064]
```

ここでは `you` が一番高く、モデルは `I love` の次に `you` がもっともありそうだと予測しています。

実際の言語モデルでは語彙サイズは数万〜十数万に及びます。バッチと系列長を含めると、logitsのshapeは次のようになります。

```text
[batch_size, seq_len, vocab_size]
例: [2, 5, 10000]
（2個の文 / 各文は5トークン / 各位置ごとに10000個の候補スコア）
```

このshapeは、言語モデル実装でとても重要です。

---

## 8.6 softmax出力を確率分布として見る

前章で、softmaxはスコアを重みに変換する関数だと説明しました。この章ではそれを確率分布として見ます。`logits = [2.0, 1.0, 0.0]`（合計3.0、まだ確率ではない）にsoftmaxをかけると `[0.665, 0.245, 0.090]`（すべて0以上、合計1）になり、確率分布として扱えます。

ここで注意したいのは、学習時にはsoftmaxを明示的に書かないことも多い点です。PyTorchの `cross_entropy` は内部で `log_softmax` を含む計算をするので、実装ではlogitsをそのまま損失関数に渡すことが多いです。

```python
loss = F.cross_entropy(logits, targets)   # logits は softmax 前のスコア
```

推論時に確率として見たい場合はsoftmaxをかけます。

```python
probs = torch.softmax(logits, dim=-1)
```

`dim=-1` は最後の次元、つまり語彙方向にかける指定です。shapeは `[batch_size, seq_len, vocab_size]` のまま変わらず、各位置ごとに語彙全体の確率の合計が1になります。

---

## 8.7 正解トークンの確率

言語モデルの学習では、正解トークンの確率が重要です。`I love dogs` という文では、途中までの文脈から次を予測します。

```text
入力: I       正解: love
入力: I love  正解: dogs
```

語彙 `["I","you","love","dogs","."]` で、`I love` の次の確率分布が次のようなら、正解 `dogs` の確率は0.70で良い予測です。

```text
I: 0.05, you: 0.10, love: 0.05, dogs: 0.70, .: 0.10
P(dogs | I love) = 0.70
```

一方 `dogs` が0.05しかなければ悪い予測です。学習では正解トークンの確率が高くなるようにモデルを更新します。

```text
正解トークンの確率が高い → 損失が小さい
正解トークンの確率が低い → 損失が大きい
```

この「正解トークンの確率」をもとに損失を計算する代表的な方法がcross entropy（次章）です。直感だけ先に言うと、正解トークンの確率が1に近いと損失が小さく、0に近いと損失が大きくなります。つまり言語モデルの学習は、単純化すれば「文脈から次トークンの確率分布を出す → 正解トークンの確率を見る → それが高くなるように重みを更新する」という流れです。

---

## 8.8 samplingとtemperature

言語モデルが文章を生成するとき、最終的に確率分布から次のトークンを選びます。一番単純なのはargmax（もっとも確率の高いトークンを選ぶ）ですが、毎回同じような出力になりやすいです。samplingでは確率に従ってランダムに選ぶので、生成に多様性が出ます。

さらに **temperature** という値で確率分布の鋭さを調整できます。softmaxに入れる前のlogitsをtemperatureで割ります。

```text
softmax(logits / temperature)
```

temperatureが低いと分布が鋭くなり（高確率の候補に集中、安定的・繰り返しやすい）、高いと平らになります（低確率の候補も選ばれやすい、多様だが破綻しやすい）。

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.0])
for t in [0.5, 1.0, 2.0]:
    print(t, torch.softmax(logits / t, dim=-1))
```

```text
0.5  tensor([0.8668, 0.1173, 0.0159])
1.0  tensor([0.6652, 0.2447, 0.0900])
2.0  tensor([0.5065, 0.3072, 0.1863])
```

temperatureが低いほど最大候補に集中し、高いほど平らになります。学習時の基本を理解する段階では「言語モデルは次トークンの確率分布を出す / 生成時にその分布から選ぶ / temperatureで鋭さを調整できる」がわかれば十分です。

---

## 8.9 PyTorchで確率分布を確認する

logitsにsoftmaxをかけると確率分布になり、argmaxやsamplingで次トークンを選べます。

```python
import torch

vocab = ["I", "you", "love", "dogs", "."]
logits = torch.tensor([0.1, 2.0, -1.0, 1.5, 0.0])
probs = torch.softmax(logits, dim=-1)

for token, prob in zip(vocab, probs):
    print(token, round(float(prob), 3))

print("argmax:", vocab[torch.argmax(probs)])
print("sample:", vocab[torch.multinomial(probs, num_samples=1)])
```

```text
I 0.071 / you 0.474 / love 0.024 / dogs 0.287 / . 0.064
argmax: you
sample: （確率に従うので you や dogs など、毎回同じとは限らない）
```

`torch.argmax` は一番確率の高いトークン、`torch.multinomial` は確率分布に従ってサンプルを選びます。

---

## 8.10 PyTorchで言語モデルの出力shapeを確認する

実際の言語モデルに近いshapeを確認します。隠れ状態 `hidden`（各位置の出力ベクトル）を語彙サイズへの線形層に通してlogitsを作り、softmaxで確率分布にします。

```python
import torch
import torch.nn as nn

batch_size, seq_len, d_model, vocab_size = 2, 4, 8, 10

hidden = torch.randn(batch_size, seq_len, d_model)
output_layer = nn.Linear(d_model, vocab_size)

logits = output_layer(hidden)
probs = torch.softmax(logits, dim=-1)

print("hidden:", hidden.shape)   # [2, 4, 8]
print("logits:", logits.shape)   # [2, 4, 10]
print("probs:", probs.shape)     # [2, 4, 10]
print("sum over vocab:", probs.sum(dim=-1))  # 全要素 1.0
```

shapeの流れは次の通りです。各位置ごとに語彙方向の確率の合計が1になり、`probs[batch, position, :]` がその位置における次トークンの確率分布です。

```text
hidden: [batch_size, seq_len, d_model]
↓ Linear(d_model, vocab_size)
logits: [batch_size, seq_len, vocab_size]
↓ softmax
probs:  [batch_size, seq_len, vocab_size]
```

この流れは、言語モデルの最後の部分（Transformerの出力 → Linear → logits → softmax → 次トークンの確率分布）に対応しています。

---

## 8.11 まとめ

この章では、確率分布について学びました。確率は、ある候補がどれくらい起こりそうかを表す値で、「0以上・1以下・すべての候補の合計が1」という性質を持ちます。確率分布とは、複数の候補に対して確率を割り当てたものです。

言語モデルでは、次のトークンを語彙全体から予測するので、最終的に語彙全体に対する確率分布を出します。Transformerの出力（各位置のベクトル `hidden: [batch_size, seq_len, d_model]`）を語彙サイズへの線形層に通すとlogits（`[batch_size, seq_len, vocab_size]`）になり、softmaxをかけると確率分布になります。

```text
probs = softmax(logits)
（shapeは変わらず、語彙方向の合計が1になる）
```

学習では正解トークンの確率が高くなるようにモデルを更新します。生成時にはこの確率分布から次トークンを選び、選び方にはargmax（一番確率の高いトークン）とsampling（確率に従ってランダム）があります。temperatureを使うと分布の鋭さ（低いと集中、高いと平ら）を調整できます。

この章で特に重要なのは、次の理解です。

```text
言語モデルは次トークンの確率分布を出す
softmaxはlogitsを確率分布に変換する
確率分布は語彙全体に対して定義される
正解トークンの確率が学習において重要である
```

次章では、損失関数とcross entropyについて学びます。cross entropyは、モデルが出した確率分布と正解トークンとのズレを数値化するために使われます。言語モデルの学習は「logitsを出す → 正解トークンと比べる → cross entropy lossを計算する → 損失が小さくなるようにパラメータを更新する」という流れが基本です。

---

<!-- chapter-nav:start -->

[← 第7章 softmax](Chapter%207%20-%20Softmax.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第9章 損失関数とクロスエントロピー →](Chapter%209%20-%20Loss%20Functions%20and%20Cross-Entropy.md)

<!-- chapter-nav:end -->
