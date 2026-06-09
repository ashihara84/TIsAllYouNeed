# 第10章　モデル全体を組む（小さな GPT）

**この章でわかること**

- embedding → ブロック列 → 出力層
- 出力を語彙上の分布にする（損失の shape）
- モデルのパラメータ数を確かめる

### 10.1　embedding → ブロック列 → 出力層

部品が、すべてそろいました。

これらを1つの `nn.Module` にまとめて、小さな GPT を、完成させます。

流れは、第5章の図のとおりです。

```text
入力 ID [B, L]
  → Embeddings（トークン + 位置）  [B, L, d_model]   （第7章）
  → Block x N（Transformer 本体）  [B, L, d_model]   （第9章）
  → 最後に LayerNorm
  → 出力層 Linear（d_model → vocab_size）  [B, L, vocab_size]
```

左から右へ、ID を入れて、ロジットを出す。

最後の出力層は、何をしているのでしょうか。

各位置の `d_model` 次元ベクトルを、語彙サイズ分のロジットに、変換する線形層です。

これで、各位置に「次の文字の、確率分布のもと（ロジット）」が、出ます（第4巻7章）。

語彙が50文字なら、各位置に50個のロジット。

「次は、この50文字のうち、どれか」を予測する、というわけです。

### 10.2　モデル全体を書く

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4,
                 n_layers=3, block_size=8, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.emb = Embeddings(vocab_size, d_model, block_size)      # 第7章
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])                                                          # 第9章
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)      # 出力層

    def forward(self, idx, targets=None):
        x = self.emb(idx)              # [B, L, d_model]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)          # [B, L, vocab_size]

        loss = None
        if targets is not None:
            B, L, V = logits.shape
            loss = F.cross_entropy(logits.view(B * L, V), targets.view(B * L))
        return logits, loss
```

`__init__` で、これまで作った部品を、すべて並べています。

- `emb`：埋め込み（第7章）。
- `blocks`：Transformer ブロック × N（第9章）。
- `ln_f`：最後の LayerNorm。
- `head`：出力層。

`forward` は、入力 ID から、各位置のロジット `[B, L, vocab_size]` を、出します。

そして、`targets`（次の文字、第6章）が渡されたら、損失も計算します。

これで、小さな GPT が、1つの `nn.Module` として、完成しました。

### 10.3　出力を語彙上の分布にする（損失の shape）

損失の計算で、`logits` と `targets` の shape を、合わせる点に、注意します。

`F.cross_entropy` は、`[N, クラス数]` のロジットと、`[N]` のターゲットを取ります。

しかし、いまの `logits` は `[B, L, vocab_size]`、`targets` は `[B, L]` の、3次元・2次元です。

そこで、バッチと系列を、まとめて平らにします（第3巻5章の `CrossEntropyLoss` の拡張）。

```text
logits  : [B, L, vocab_size]  →  view  →  [B*L, vocab_size]
targets : [B, L]              →  view  →  [B*L]
```

`view(B * L, V)` で、`[B, L, vocab_size]` を `[B*L, vocab_size]` に、平らにします。

たとえば `[4, 8, 50]` なら、`[32, 50]` に。

「4区間 × 8位置 = 32個の予測」を、まとめて1つの損失に、するのです。

これで、全位置・全バッチの「次の文字予測」を、まとめて1つの損失（スカラー）に、できます。

第4巻7章で「各位置が巨大な分類問題」と述べた、その損失です。

### 10.4　モデルのパラメータ数を確かめる

組み上がったモデルの、パラメータ数を、数えてみましょう。

第5章で触れた `parameters()` で、集計できます。

```python
import torch

torch.manual_seed(0)
model = SmallGPT(vocab_size=50, d_model=64, n_heads=4, n_layers=3, block_size=8)

n_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数: {n_params:,}")

# 順伝播の確認
idx = torch.randint(0, 50, (4, 8))
logits, loss = model(idx, targets=torch.randint(0, 50, (4, 8)))
print(logits.shape, loss.item())   # [4, 8, 50], スカラーの損失
```

GPT という名前ですが、これは数万〜十数万パラメータの、ごく小さなモデルです。

本物の GPT は、数十億〜数千億パラメータです。

その差は、とても大きいです。

しかし、構造は、本質的に同じ――embedding ＋ Transformer ブロック ＋ 出力層――です。

違うのは、規模だけ。

第8巻で、この小さな GPT と、本物の違いを、回収します。

モデルが完成し、順伝播と損失が、通りました。

次章で、これを実際に、学習させます。

### 10.5　本章のまとめ

- 小さな GPT は「Embeddings（第7章）→ Block × N（第9章）→ LayerNorm → 出力層 Linear」。
- 出力は各位置のロジット `[B, L, vocab_size]`。次の文字の分布のもと（第4巻7章）。
- 損失は `logits` と `targets` を `[B*L, ...]` に平らにして cross_entropy で計算する（例：`[4,8,50]`→`[32,50]`）。
- 構造は本物の GPT と本質的に同じ。違うのは規模だけ（第8巻で回収）。次章で学習させる。
