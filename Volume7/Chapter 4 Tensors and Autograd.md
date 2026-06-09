# 第4章　テンソルと autograd

**この章でわかること**

- テンソルの基本操作の確認
- requires_grad と計算グラフ
- 勾配の取り出しと zero_grad
- no_grad と detach（第2巻12章の回収）

この章は、これから小さな GPT を組むために必要な、PyTorch の道具を、短く整理します。

多くは、第2巻・第3巻で触れたものの、復習です。

道具箱の中身を、もう一度、確認しておく、という感じです。

### 4.1　テンソルの基本操作

テンソルは、NumPy 配列によく似た、多次元配列です。

GPU でも動き、autograd に対応しています。

第2巻2章で学んだ shape の感覚が、そのまま効きます。

```python
import torch

x = torch.randn(2, 3, 4)      # [batch=2, L=3, d=4]
print(x.shape)                # torch.Size([2, 3, 4])
print(x.mean(dim=-1).shape)   # torch.Size([2, 3])  最後の次元で平均
print((x @ x.transpose(-2, -1)).shape)   # [2, 3, 3]  バッチ行列積
```

一つずつ、確認しましょう。

`x.shape` は、テンソルの形です。

`x.mean(dim=-1)` は、最後の次元で平均を取るので、その次元が消えて `[2, 3]` になります。

`x @ x.transpose(-2, -1)` は、バッチ付きの行列積です。

`@`（行列積）、`transpose`、`mean(dim=...)` など、第5巻で Attention を書くのに使った操作は、すべて、テンソルの基本操作です。

第5巻で何度も使ったので、もう手に馴染んでいるはずです。

### 4.2　requires_grad と計算グラフ

`requires_grad=True` のテンソルに対する演算は、PyTorch が **計算グラフ** として、記録します（第2巻12章）。

計算グラフとは、何でしょうか。

「どの値が、どの演算で、どの値から作られたか」の、記録です。

`backward()` のときに、これを逆向きにたどって、勾配を計算します。

簡単な例で、確かめましょう。

```python
import torch

w = torch.tensor(3.0, requires_grad=True)
y = w * w + 2 * w        # 計算が記録される
y.backward()             # dy/dw を計算
print(w.grad)            # tensor(8.)  （2w + 2 = 8）
```

`y = w² + 2w` を、`w` で微分すると、`2w + 2` です。

`w = 3` なら、`2×3 + 2 = 8`。

`w.grad` に、ちゃんと `8` が入っています。

手で微分した答えと、一致します。

`nn.Linear` や `nn.Embedding` のパラメータは、最初から `requires_grad=True` です。

だから、それらを使って組んだモデルは、`loss.backward()` だけで、全パラメータの勾配が求まります。

### 4.3　勾配の取り出しと zero_grad

勾配は、各パラメータの `.grad` に、入ります。

そして、重要なのは、**勾配は加算されていく** ことです（第3巻6章で学びました）。

何もしないと、前回の勾配に、足され続けます。

だから、各ステップの前に、消す必要があります。

```python
# 自分でパラメータを持つ場合
p.grad.zero_()

# optimizer を使う場合（後の章ではこちら）
optimizer.zero_grad()
```

`optimizer.zero_grad()` は、optimizer に登録した全パラメータの勾配を、一括で消します。

学習ループでは、毎ステップ、これを呼びます。

「消す → 計算する → 更新する」の、最初の「消す」が、これでした（第3巻7章）。

### 4.4　no_grad と detach（第2巻12章の回収）

学習しないとき（推論・生成・評価）には、計算グラフを作る必要が、ありません。

`torch.no_grad()` で囲むと、グラフ記録を止めて、メモリと計算を、節約できます（第2巻12章、第3巻8章で使いました）。

```python
with torch.no_grad():
    logits = model(x)      # 勾配を追跡しない（推論・生成で使う）
```

推論や生成のときは、勾配は要りません。

ただ、モデルに通して、答えを得るだけです。

だから `no_grad` で囲んで、無駄を省きます。

`detach()` は、あるテンソルを、計算グラフから切り離します。

「ここから先は、勾配を流さない」としたいときに、使います。

生成ループで、出力を次の入力に回すときなどに、役立ちます。

これで、小さな GPT を組むための、道具がそろいました。

次章から、第5巻の部品を、`nn.Module` として組み直していきます。

### 4.5　本章のまとめ

- テンソルは autograd 対応の多次元配列。`@`・`transpose`・`mean(dim=...)` など、第5巻で使った操作が効く。
- `requires_grad=True` の演算は計算グラフに記録され、`backward()` で勾配が求まる（`w²+2w` の例で確認）。`.grad` に入る。
- 勾配は加算されるので、毎ステップ `zero_grad()` で消す。
- 推論・生成では `torch.no_grad()` でグラフ記録を止める。`detach()` はグラフからの切り離し。

---

<!-- chapter-nav:start -->

[← 第3章　同じネットを PyTorch で書き直す](Chapter%203%20Rewriting%20in%20PyTorch.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第5章　nn.Module で部品を書く →](Chapter%205%20Writing%20Parts%20as%20nn%20Module.md)

<!-- chapter-nav:end -->
