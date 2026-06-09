# 第7章　埋め込みと位置エンコーディングを組む

**この章でわかること**

- トークン埋め込み層
- 位置エンコーディングを足す（第5巻8〜9章の回収）
- 入力表現を確かめる（shape）

### 7.1　トークン埋め込み層

第6章で、データは `[B, L]` の文字 ID に、なりました。

これを `[B, L, d_model]` のベクトル列にするのが、トークン埋め込みです（第4巻5章）。

`nn.Embedding` を、使います。

```python
import torch
import torch.nn as nn

vocab_size = 50    # 第6章の語彙サイズ（例）
d_model = 64

tok_emb = nn.Embedding(vocab_size, d_model)

x_ids = torch.randint(0, vocab_size, (4, 8))   # [B=4, L=8]
tok = tok_emb(x_ids)
print(tok.shape)   # torch.Size([4, 8, 64])  [B, L, d_model]
```

`nn.Embedding` が、各文字 ID を、`d_model` 次元のベクトルに、変換します。

第4巻5章で確かめた「行を引く操作」です。

ID `[4, 8]` を渡すと、`[4, 8, 64]`（各文字がベクトルに）が返ります。

### 7.2　位置エンコーディングを足す（第5巻8〜9章の回収）

Attention は順序を知らないので、位置の情報を足す必要が、ありました（第5巻8章）。

第1章で述べたとおり、この巻では、実装が素直な **学習する位置埋め込み**（第5巻9章）を、使います。

なぜ sin/cos でなく、学習版なのか。

学習版のほうが、実装が単純で、固定長の学習に向くからです（第5巻8〜9章で、両方を実装し、違いを押さえました）。

位置 0〜L-1 の、それぞれにベクトルを学習し、トークン埋め込みに、足します。

```python
import torch
import torch.nn as nn

max_len = 128
pos_emb = nn.Embedding(max_len, d_model)   # 位置 -> ベクトル

def add_positions(tok):
    B, L, _ = tok.shape
    pos = torch.arange(L, device=tok.device)    # [L]
    return tok + pos_emb(pos)                    # [B, L, d_model]（ブロードキャスト）

x = add_positions(tok)
print(x.shape)   # torch.Size([4, 8, 64])
```

`torch.arange(L)` で、位置の番号 `[0, 1, ..., L-1]` を作ります。

`pos_emb(pos)` で、各位置のベクトルを引きます（`[L, d_model]`）。

これを、`[B, L, d_model]` のトークン埋め込みに、足します。

ブロードキャストで、全バッチに、同じ位置ベクトルが、加わります。

これで、同じ文字でも、位置によって、少し違う表現になります。

Attention が、順序を扱えるようになりました（第5巻8章）。

### 7.3　入力表現を確かめる（shape）

トークン埋め込みと位置埋め込みを、まとめて、モデルの入口を `nn.Module` にしておきます。

```python
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, idx):
        # idx: [B, L]
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        return self.tok_emb(idx) + self.pos_emb(pos)   # [B, L, d_model]

emb = Embeddings(vocab_size=50, d_model=64, max_len=128)
out = emb(torch.randint(0, 50, (4, 8)))
print(out.shape)   # torch.Size([4, 8, 64])
```

`forward` の中で、トークン埋め込み（`tok_emb`）と位置埋め込み（`pos_emb`）を、足しています。

shape の流れを、追いましょう。

```text
idx : [4, 8]        （4区間、各8文字の ID）
 ↓ tok_emb
    : [4, 8, 64]    （各文字がベクトルに）
 ↓ + pos_emb
out : [4, 8, 64]    （位置の情報が加わる）
```

入力 `[B, L]` が、`[B, L, d_model]` の入力表現に、なりました。

これが、次章以降の Transformer ブロックへ、流れていく入口です。

第4巻6章で「embedding しただけでは各トークンは独立」と、述べました。

次章の Attention が、ここに、文脈を混ぜ込みます。

### 7.4　本章のまとめ

- `nn.Embedding` で、文字 ID `[B, L]` を、トークン埋め込み `[B, L, d_model]` にする（第4巻5章）。
- 位置の情報は、学習する位置埋め込みを足して与える（第5巻9章）。実装が素直なので、この巻ではこちらを使う。
- トークン埋め込み＋位置埋め込みを、`Embeddings` モジュールにまとめた。出力は `[B, L, d_model]`。
- これがモデルの入口。文脈を混ぜるのは、次章の Attention。

---

<!-- chapter-nav:start -->

[← 第6章　データを用意する（文字レベル）](Chapter%206%20Preparing%20Character%20Level%20Data.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第8章　Attention 部品を nn.Module にする →](Chapter%208%20Attention%20as%20nn%20Module.md)

<!-- chapter-nav:end -->
