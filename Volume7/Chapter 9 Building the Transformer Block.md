# 第9章　Transformer ブロックを組む

**この章でわかること**

- Attention + FFN + 残差 + LayerNorm（第5巻10〜11章の回収）
- ブロックを `nn.Module` にまとめる
- Pre-LN を採用する理由
- ブロックを積む

### 9.1　FFN を用意する

ブロックには、Attention のほかに、FFN が要ります（第5巻10章）。

文字レベル GPT 用に、Dropout を足して、用意します。

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # 広げる（第5巻10章：4倍が定番）
            nn.GELU(),                          # ReLU の親戚（第4巻4章で触れた）
            nn.Linear(4 * d_model, d_model),    # 戻す
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

中間で4倍に広げて、戻す。

位置ごとの2層 MLP です（第5巻10章）。

`d_model` を `4 * d_model` に広げ、活性化を通し、`d_model` に戻す。

活性化は、現代の Transformer でよく使われる GELU にしています（ReLU でも動きます。第4巻4章）。

### 9.2　Pre-LN を採用する理由

第5巻11章で、`LayerNorm(x + Sublayer(x))` の Post-LN（論文）と、`x + Sublayer(LayerNorm(x))` の Pre-LN（現代）の、違いを学びました。

この巻では、**Pre-LN** を、採用します。

```text
Post-LN（論文）: x = LayerNorm(x + Sublayer(x))
Pre-LN（本書）  : x = x + Sublayer(LayerNorm(x))   ← こちらを使う
```

なぜ Pre-LN なのでしょうか。

Pre-LN のほうが、深く積んでも学習が安定しやすく、学習率のウォームアップなしでも、動きやすいからです。

小さな GPT を、手軽に学習させるのに、向いています。

論文は Post-LN でしたが、現代の多くの実装は、Pre-LN です。

第5巻11章で「第7巻では Pre-LN を使う」と予告した、その通りです。

### 9.3　ブロックを `nn.Module` にまとめる

第5巻11章の `TransformerBlock` を、Pre-LN で書きます。

Attention（第8章）と FFN（本章）を、それぞれ LayerNorm をかけてから、残差で足します。

```python
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Pre-LN：先に LayerNorm、残差で足す
        x = x + self.ffn(self.ln2(x))
        return x
```

`forward` の2行を、よく見てください。

```text
x = x + self.attn(self.ln1(x))   ← x + Attention(LayerNorm(x))
x = x + self.ffn(self.ln2(x))    ← x + FFN(LayerNorm(x))
```

`x + self.attn(self.ln1(x))` が、Pre-LN の `x + Sublayer(LayerNorm(x))` です。

先に `ln1` で LayerNorm をかけ、`attn` を通し、生の `x` を足す。

第3巻9〜10章で仕込んだ残差接続と LayerNorm、第5巻で作った Attention と FFN。

すべてが、この数行に、集まっています。

第3巻で「中身は第5巻で」と言った `Sublayer` が、ここで、Attention と FFN として、完全に埋まりました。

### 9.4　ブロックを積む

ブロックは、入出力 `[B, L, d_model]` が同じなので、何段でも積めます（第5巻12章）。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
blocks = nn.ModuleList([
    Block(d_model=64, n_heads=4, block_size=8) for _ in range(3)
])

x = torch.randn(4, 8, 64)
for blk in blocks:
    x = blk(x)
print(x.shape)   # torch.Size([4, 8, 64])
```

3段積んでも、shape は不変。

`nn.ModuleList` は、複数のモジュールをリストとして持つ仕組みで、各ブロックのパラメータも、ちゃんと登録されます（第5章）。

段数（`n_layers`）は、モデルの深さを決める、ハイパーパラメータです。

これで、Transformer の本体（ブロックの積み重ね）が、できました。

次章で、これを embedding（第7章）と出力層で挟んで、モデル全体（小さな GPT）にします。

### 9.5　本章のまとめ

- ブロックは Attention（第8章）＋ FFN（本章）を、LayerNorm と残差で包んだもの（第5巻10〜11章）。
- この巻は Pre-LN（`x + Sublayer(LayerNorm(x))`）を採用。深く積んでも安定しやすく、手軽に学習できる。
- 第3巻の残差・LayerNorm と、第5巻の Attention・FFN が、`Block` の数行に集約された（`Sublayer` が埋まった）。
- ブロックは入出力 shape が同じなので、何段でも積める（`nn.ModuleList`）。段数はハイパーパラメータ。
