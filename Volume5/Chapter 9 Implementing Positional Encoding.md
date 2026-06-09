# 第9章　位置エンコーディングを書く

**この章でわかること**

- sin / cos の位置エンコーディングを実装する
- embedding に足して、確かめる
- 学習する位置埋め込みも、書いてみる

### 9.1　sin / cos の位置エンコーディングを実装する

前章の式を、そのままコードにします。

位置 `pos`（0 から L-1）と、次元 `i` について、偶数次元は sin、奇数次元は cos でした。

```python
import torch
import math

def sinusoidal_pe(L, d_model):
    pe = torch.zeros(L, d_model)
    pos = torch.arange(L).unsqueeze(1)           # [L, 1]
    # 10000^(2i/d_model) を div_term としてまとめて計算
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )                                            # [d_model/2]
    pe[:, 0::2] = torch.sin(pos * div_term)      # 偶数次元
    pe[:, 1::2] = torch.cos(pos * div_term)      # 奇数次元
    return pe                                     # [L, d_model]

pe = sinusoidal_pe(L=10, d_model=16)
print(pe.shape)        # torch.Size([10, 16])
print(pe[0, :4])       # 位置0
print(pe[1, :4])       # 位置1（位置0と違う値になる）
```

コードを、少し読み解きましょう。

`pe` は、`[L, d_model]` の入れ物（最初は全部0）です。

`pos` は、位置の番号（0, 1, 2, ...）を縦に並べたものです。

`div_term` は、次元ごとの「波の速さ」を決める値です（前章の `10000^(2i/d_model)` の部分）。

`pe[:, 0::2]` は「偶数番目の列」、`pe[:, 1::2]` は「奇数番目の列」を指します。

偶数列に sin、奇数列に cos を入れています。

`pe[pos]` が、位置 `pos` の指紋ベクトルです。

実行すると、`pe[0]`（位置0）と `pe[1]`（位置1）は、違う値になります。

位置が違えば、異なるベクトルになるのです。

しかも、`-1〜1` の範囲の、なめらかな値です。

この関数は、学習パラメータを持たず、計算だけで決まります（固定の位置エンコーディング）。

何回呼んでも、同じ位置には、同じベクトルが返ります。

### 9.2　embedding に足して確かめる

トークンの embedding（第4巻5章）に、この位置エンコーディングを、足します。

shape が同じ `[L, d_model]` なので、そのまま足せます。

```python
import torch
import torch.nn as nn

vocab_size, d_model, L = 100, 16, 10
emb = nn.Embedding(vocab_size, d_model)

ids = torch.randint(0, vocab_size, (1, L))   # [batch=1, L]
x = emb(ids)                                  # [1, L, d_model]

pe = sinusoidal_pe(L, d_model)                # [L, d_model]
x = x + pe                                     # ブロードキャストで [1, L, d_model]
print(x.shape)                                # torch.Size([1, 10, 16])
```

`x + pe` で、各トークンの embedding に、その位置の指紋が、加わります。

`x` が `[1, 10, 16]`、`pe` が `[10, 16]` ですが、ブロードキャストで、バッチの各文に、同じ位置ベクトルが足されます。

これで、同じトークンでも、位置によって、少し違うベクトルになりました。

Attention が、順序を扱えるようになったのです。

この `x` が、Attention に渡る、最終的な入力表現です。

### 9.3　学習する位置埋め込みも書いてみる

前章で触れた、学習する位置埋め込みも、書いておきましょう。

`nn.Embedding` を「位置の番号 → ベクトル」として使うだけです。

```python
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [batch, L, d_model]
        L = x.size(1)
        positions = torch.arange(L, device=x.device)   # [L]
        return x + self.pos_emb(positions)             # 位置ベクトルを足す

pe_layer = LearnedPositionalEmbedding(max_len=512, d_model=16)
x = torch.randn(2, 10, 16)
print(pe_layer(x).shape)    # torch.Size([2, 10, 16])
```

トークン埋め込みとの違いは、引くものが「単語の ID」ではなく「位置の番号」だ、という点だけです。

`torch.arange(L)` で、位置の番号 `[0, 1, ..., L-1]` を作り、それを `nn.Embedding` に渡して、位置ベクトルを引きます。

`pos_emb` の中身は、学習で決まるパラメータです。

sin/cos と違って学習しますが、`max_len`（ここでは512）を超える位置は、扱えません。

512個分の位置ベクトルしか、持っていないからです。

第7巻で小さな GPT を作るときは、この学習する位置埋め込みを使います（実装が素直で、固定長の学習に向くため）。

この巻では、「論文の sin/cos」と「現代でよく使う学習版」の、両方を手に入れました。

### 9.4　本章のまとめ

- sin/cos の位置エンコーディングは、位置と次元から計算だけで決まる、固定のベクトル。`[L, d_model]`。
- 偶数次元に sin、奇数次元に cos を入れる。`pe[pos]` が、位置ごとの「指紋」。
- トークンの embedding に足す（shape が同じなので、そのまま `x + pe`）。これで Attention が順序を扱える。
- 学習する位置埋め込みは `nn.Embedding` を位置に使うだけ。柔軟だが `max_len` まで。第7巻ではこちらを使う。

---

<!-- chapter-nav:start -->

[← 第8章　位置エンコーディング](Chapter%208%20Positional%20Encoding.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第10章　Feed-Forward Network（FFN） →](Chapter%2010%20Feed-Forward%20Network.md)

<!-- chapter-nav:end -->
