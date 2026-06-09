# 第5章　Self-Attention

**この章でわかること**

- 自分自身を Q・K・V にする、とはどういうことか
- 各トークンが系列全体を見る、という意味
- Self-Attention モジュールを `nn.Module` として書く
- 出力の shape と意味を、確かめる

### 5.1　自分自身を Q・K・V にするとは

前章の `attention(Q, K, V)` は、Q・K・V がどこから来たかを、問いませんでした。

3つのベクトルを受け取って、計算するだけでした。

**Self-Attention**（自己注意）とは、同じ入力系列から、Q・K・V の3つすべてを作る、という使い方です。

```text
入力 x（系列）
Q = x W_Q
K = x W_K
V = x W_V
出力 = Attention(Q, K, V)
```

入力 `x` 1つから、Q も K も V も作る。

だから「自分（self）の中で、自分の各トークンが、自分の各トークンに注目する」ので、Self-Attention です。

第3章で Q/K/V を `x` から作りましたが、まさにあれが、Self-Attention の入り口でした。

念のため、対比を出しておきます。

Self-Attention に対して、Q を一方の系列から、K・V を別の系列から作る使い方を、Cross-Attention と呼びます。

たとえば翻訳で、出力（日本語）の Query が、入力（英語）の Key・Value を見る、というような使い方です。

Cross-Attention は、第6巻で論文を読むときに、出てきます。

この巻で作るのは、Self-Attention です。

### 5.2　各トークンが系列全体を見る

Self-Attention では、何が起きるでしょうか。

系列内の各トークンが、自分を含む系列全体の、全トークンを見渡します。

そして、相性に応じて、情報を集めます。

具体例で、イメージしましょう。

```text
"犬 が 走る" の Self-Attention:
  「走る」は「犬」（主語）に強く注目して、その情報を取り込む
  「が」は構文的な役割で、周りに注目する
  ...各トークンが、文脈の必要な相手から、情報を集める
```

「走る」というトークンを処理するとき、「誰が走るのか」を知るために、「犬」に注目する。

そして、「犬」の情報を取り込んで、「（犬が）走る」という、文脈を含んだ表現に、自分を更新する。

第4巻6章で「embedding しただけでは各トークンは独立」と述べました。

Self-Attention を通すと、各トークンの表現が、文脈中の関連トークンの情報を取り込んだものへと、更新されます。

入力 `[L, d_model]` を受け取り、文脈を混ぜた `[L, d_model]` を返す。

形は同じで、中身が豊かになるのです。

### 5.3　Self-Attention モジュールを書く

第3巻8章で学んだ `nn.Module` の作法で、Self-Attention を、1つの部品として書きます。

`__init__` で Q/K/V の線形変換を用意し、`forward` で「Q/K/V を作る → attention」を行います。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        # x: [batch, L, d_model]
        Q = self.W_q(x)        # [batch, L, d_k]
        K = self.W_k(x)
        V = self.W_v(x)

        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # [batch, L, L]
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                   # [batch, L, d_k]
        return out
```

第3巻8章の `nn.Module` の型（`__init__` で部品、`forward` で流れ）を、思い出してください。

ここでは、

- `__init__` で、Q/K/V の3つの線形変換を用意し、
- `forward` で、Q/K/V を作って、前章の attention の計算をする、

という形になっています。

第4章の `attention` 関数を、`nn.Module` の中に取り込んだ形です。

Q/K/V を作る重みを、部品として持つので、これは学習可能な、1つのまとまり（モジュール）になりました。

### 5.4　出力の shape と意味を確かめる

動かして、shape を確認しましょう。

```python
import torch

torch.manual_seed(0)
d_model, d_k = 16, 16
sa = SelfAttention(d_model, d_k)

x = torch.randn(2, 5, d_model)   # [batch=2, L=5, d_model=16]
out = sa(x)
print(out.shape)                 # torch.Size([2, 5, 16])
```

入力 `[2, 5, 16]` に対して、出力も `[2, 5, 16]` です（`d_k = d_model` にした場合）。

shape の変化を、確認しましょう。

```text
入力 x : [2, 5, 16]   （2文、各5トークン、各16次元）
 ↓ Self-Attention
出力   : [2, 5, 16]   （形は同じ。中身が文脈を取り込んで更新）
```

系列長 `5` も保たれ、各トークンの位置はそのまま。

各位置の表現だけが、文脈を取り込んで、更新されています。

この「入力と出力の shape が同じ」という性質は、とても重要です。

なぜでしょうか。

だからこそ、Self-Attention を、第3巻の残差接続 `x + Sublayer(x)` で包めるのです。

第3巻9章で「足し算には同じ shape が必要」と学びました。

入出力が同じ `[2, 5, 16]` なら、`x + Sublayer(x)` がちゃんと足せます。

そして、ブロックを何段も、積み重ねられます。

`Sublayer` の中身が、ついに Self-Attention で、埋まりつつあります。

ただし、いまの Self-Attention には、1つ足りないものがあります。

各トークンが「系列全体」を見る、つまり **未来のトークンも見てしまう** のです。

「犬 が 走る」で、「が」を処理するとき、いまの Self-Attention は、未来の「走る」も見られてしまいます。

言語モデルでは、次の単語を予測するのに、その答え（未来）を見てはいけません。

これを禁じるのが、次章の causal mask です。

### 5.5　本章のまとめ

- Self-Attention は、同じ入力 `x` から Q・K・V すべてを作る使い方。各トークンが系列全体に注目する。
- 入力 `[batch, L, d_model]` を受け取り、文脈を混ぜた同じ shape を返す。形は不変、中身が豊かになる。
- `nn.Module` として、Q/K/V の線形変換を持ち、`forward` で第4章の attention を行う部品にした。
- 入出力の shape が同じなので、残差接続で包め、ブロックを積み重ねられる。
- いまの Self-Attention は未来のトークンも見てしまう。それを禁じるのが、次章の causal mask。

---

<!-- chapter-nav:start -->

[← 第4章　スケール付き内積 Attention](Chapter%204%20Scaled%20Dot-Product%20Attention.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第6章　Causal mask（未来を見ない） →](Chapter%206%20Causal%20Masking.md)

<!-- chapter-nav:end -->
