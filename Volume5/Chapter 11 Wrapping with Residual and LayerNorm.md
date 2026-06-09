# 第11章　残差接続と LayerNorm で部品を包む

**この章でわかること**

- 第3巻で仕込んだ残差接続と LayerNorm の回収
- `LayerNorm(x + Sublayer(x))` の形
- Attention と FFN を、それぞれ包む
- Pre-LN と Post-LN の違い（直感）

この章で、ついに第3巻の約束が、果たされます。

`LayerNorm(x + Sublayer(x))` の `Sublayer` の中身が、Attention と FFN で、埋まります。

第3巻からずっと保留にしてきた、あの `Sublayer` が、ここで正体を現すのです。

### 11.1　第3巻で仕込んだものの回収

第3巻9章で残差接続 `x + f(x)`、10章で LayerNorm、そして両者を組み合わせた `LayerNorm(x + Sublayer(x))` の骨格を、実装しました。

あのとき、`Sublayer` の中身は、まだ「ただの線形＋ReLU」でした。

仮の中身だったのです。

```text
第3巻：  LayerNorm(x + Sublayer(x))    Sublayer = 線形+ReLU（仮）
この巻：  Sublayer = Multi-Head Attention（第7章）または FFN（第10章）
```

ここまでで作った Attention も FFN も、入出力の shape が、`[batch, L, d_model]` で **同じ** でした。

これは、偶然ではありません。

残差接続 `x + Sublayer(x)` の足し算ができるよう、意図的にそろえてきたものです。

第5章でも、第7章でも、第10章でも、「入出力は `[B, L, d_model]` で不変」と、繰り返し確認しました。

その積み重ねが、ここで効きます。

だから、Attention も FFN も、そのまま殻に包めるのです。

### 11.2　`LayerNorm(x + Sublayer(x))` の形

包み方は、第3巻10章で書いたとおりです。

サブlayer（Attention か FFN）の出力に、入力を足し（残差接続）、LayerNorm で整えます。

```text
出力 = LayerNorm(x + Sublayer(x))
```

一つずつ、役割を確認しましょう。

- `x +` は残差接続。勾配のバイパスを作り、深く積んでも学習できるようにする（第3巻9章）。
- `LayerNorm(...)` は値のスケールを整え、学習を安定させる（第3巻10章）。
- `Sublayer` は Attention か FFN。

この形のおかげで、Attention や FFN を何段も積み重ねても、勾配が流れ、スケールが暴れません。

第3巻で仕込んだ2つの道具（残差接続と LayerNorm）が、ここで、Transformer を深く積むことを、可能にしているのです。

### 11.3　Attention と FFN をそれぞれ包む

Transformer のブロックは、「包んだ Attention」と「包んだ FFN」を、順に通したものです。

それぞれを、残差＋LayerNorm で包みます。

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)   # 第7章
        self.ffn = FeedForward(d_model, d_ff)              # 第10章
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # サブlayer1：Attention を残差＋LayerNorm で包む
        x = self.norm1(x + self.attn(x))
        # サブlayer2：FFN を残差＋LayerNorm で包む
        x = self.norm2(x + self.ffn(x))
        return x

block = TransformerBlock(d_model=16, n_heads=4)
x = torch.randn(2, 5, 16)
print(block(x).shape)   # torch.Size([2, 5, 16])
```

`forward` の2行を、よく見てください。

```text
x = self.norm1(x + self.attn(x))   ← LayerNorm(x + Attention(x))
x = self.norm2(x + self.ffn(x))    ← LayerNorm(x + FFN(x))
```

1行目の `self.norm1(x + self.attn(x))` が、まさに `LayerNorm(x + Sublayer(x))` です。

`Sublayer` が、Attention になりました。

2行目の `self.norm2(x + self.ffn(x))` は、`Sublayer` が FFN です。

第3巻の `Sublayer` が、ついに、Attention と FFN で、完全に埋まりました。

第3巻から「中身は第5巻で」と保留してきたものが、いま、姿を現したのです。

入出力は、やはり `[2, 5, 16]` で不変。

だから、このブロックを、何段も積み重ねられます。

それが Transformer です。

### 11.4　Pre-LN と Post-LN の違い（直感）

上の実装は、論文のオリジナルの形で、**Post-LN** と呼ばれます。

残差を足した「後」に LayerNorm をかけるからです。

```text
Post-LN（論文）: x = LayerNorm(x + Sublayer(x))
Pre-LN（現代）  : x = x + Sublayer(LayerNorm(x))
```

現代の実装では、**Pre-LN** が、よく使われます。

サブlayer に入れる「前」に LayerNorm をかけ、残差は生の `x` をそのまま足す形です。

2つの違いは、LayerNorm を「いつかけるか」だけです。

```text
Post-LN: 残差を足してから、LayerNorm （論文）
Pre-LN : LayerNorm してから、サブlayer。残差は生の x  （現代）
```

Pre-LN は、深く積んだときに、学習が安定しやすい（勾配がより素直に流れる）ことが知られています。

```python
# Pre-LN 版の forward
def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    return x
```

どちらも「残差接続＋LayerNorm でサブlayer を包む」点は、同じです。

LayerNorm の位置が、違うだけです。

第6巻で論文を読むときは、Post-LN が出てきます。

第7巻で小さな GPT を作るときは、安定性から Pre-LN を採用します。

いまは、違いを直感で押さえておけば、十分です。

### 11.5　本章のまとめ

- これまで作った Attention と FFN は、入出力 shape が同じ。だから残差接続で、そのまま包める（意図的にそろえてきた）。
- ブロックは「残差＋LayerNorm で包んだ Attention」→「残差＋LayerNorm で包んだ FFN」の2段。
- `LayerNorm(x + Sublayer(x))` の `Sublayer` が、ついに Attention と FFN で埋まった（第3巻の回収完了）。
- Post-LN（論文）と Pre-LN（現代）の違いは、LayerNorm の位置だけ。第7巻では Pre-LN を使う。
