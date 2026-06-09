# 第8章　Attention 部品を nn.Module にする

**この章でわかること**

- Scaled Dot-Product Attention の確認（第5巻4章の回収）
- causal mask を組み込む（第5巻6章）
- Multi-Head Attention を `nn.Module` 化（第5巻7章）
- register_buffer とは何か
- 動作と shape を確かめる

### 8.1　第5巻の Attention を持ってくる

この章は、ほとんど、第5巻の回収です。

第5巻7章で書いた、causal mask 付き Multi-Head Attention を、文字レベル GPT 用に、少しだけ整えて、使います。

確認しておきましょう。

- 中心式は `softmax(QK^T / sqrt(d_k)) V`（第5巻4章）。
- 未来を隠す causal mask（第5巻6章）。
- ヘッド分割（第5巻7章）。

すべて、手の内です。

新しい概念は、ありません。

第5巻で作ったものを、統合用に、少し整えるだけです。

### 8.2　causal mask 付き Multi-Head Attention

GPT は Decoder-only で、未来を見てはいけません（第6巻4章）。

だから causal mask を、必ず使います。

第5巻7章のコードに、`block_size` 分の mask を、あらかじめ用意しておく実装にします。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # Q,K,V をまとめて
        self.proj = nn.Linear(d_model, d_model, bias=False)     # W_O
        self.dropout = nn.Dropout(dropout)
        # 下三角の causal mask を用意（学習対象でないので buffer に）
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)                      # [B, L, 3C]
        q, k, v = qkv.split(C, dim=-1)         # それぞれ [B, L, C]

        # ヘッドに分割： [B, L, C] -> [B, h, L, d_k]
        def split(t): return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        att = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)      # [B, h, L, L]
        att = att.masked_fill(self.mask[:L, :L] == 0, float("-inf"))  # 未来を隠す
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v                          # [B, h, L, d_k]

        out = out.transpose(1, 2).contiguous().view(B, L, C)   # ヘッド結合
        return self.proj(out)                  # [B, L, d_model]
```

第5巻7章のコードと、見比べてください。

違いは、2点だけです。

**1. Q/K/V を、3つの `nn.Linear` でなく、1つにまとめた。**

`self.qkv = nn.Linear(d_model, 3 * d_model)` で、Q・K・V をまとめて作り、`split` で3つに分けています。

これは、効率化のための、よくある書き方です（本質は同じ）。

**2. mask を `register_buffer` で持ち、Dropout を足した。**

mask を毎回作らず、最初に1回だけ作って、持っておきます。

Dropout は、過学習を抑えるため（第3巻11章。論文も Attention に Dropout を使う、第6巻9章）。

本質――内積・スケール・mask・softmax・重み付き和・ヘッド結合――は、第5巻のままです。

### 8.3　register_buffer とは

`register_buffer` は、何でしょうか。

「モデルに属するが、学習しないテンソル」を、登録する仕組みです。

causal mask は、固定の三角行列で、学習で変わりません。

だが、モデルと一緒に GPU へ移動したり、保存したりは、したい。

そういうものを、buffer として持ちます。

区別を、はっきりさせましょう。

```text
パラメータ（parameter）: 学習する。勾配が計算され、更新される（重みなど）。
バッファ（buffer）     : 学習しない。だがモデルに属する（mask など）。
```

mask は、学習する重みでは、ありません。

ただの固定の道具です。

でも、モデルの一部として、扱いたい。

だから、`register_buffer` で登録するのです。

この区別は、覚えておくと、役立ちます。

### 8.4　動作と shape を確かめる

動かして、shape を確認しましょう。

```python
import torch

torch.manual_seed(0)
mha = MultiHeadAttention(d_model=64, n_heads=4, block_size=8)

x = torch.randn(4, 8, 64)    # [B, L, d_model]
out = mha(x)
print(out.shape)             # torch.Size([4, 8, 64])
```

入力 `[4, 8, 64]` に対して、出力も `[4, 8, 64]`。

shape が保たれるので、残差接続で包め、ブロックを積めます（次章）。

第5巻で「部品を作る」段階だったものが、いま「統合に使える部品」として、手元にあります。

第5巻でていねいに「入出力は不変」と確認してきたことが、ここで、つなぐだけで動く、という形で、報われます。

### 8.5　本章のまとめ

- この章は第5巻4・6・7章の回収。causal mask 付き Multi-Head Attention を、GPT 用に整えた。
- Q/K/V を1つの `nn.Linear` にまとめ、Dropout を足し、mask を `register_buffer` で持つ実装にした（本質は第5巻のまま）。
- `register_buffer` は「学習しないがモデルに属するテンソル」（mask など）を登録する仕組み。パラメータとは区別する。
- 入出力 `[B, L, d_model]` は不変。残差で包め、ブロックに積める（次章）。
