# 第11章　学習ループを書く

**この章でわかること**

- 損失（cross entropy）をつなぐ
- optimizer（AdamW）と学習率
- 学習ループを回し、損失が下がるのを確認する
- よくある詰まり

### 11.1　学習ループの骨格は変わらない

第3巻7章で学んだ学習ループの骨格――forward → loss → backward → step――は、モデルが小さな GPT になっても、まったく同じです。

これが、第3巻でループの形を、身につけておいた効果です。

XOR のときも、小さな GPT のときも、骨格は、寸分変わりません。

```text
1. get_batch でバッチを取る（第6章）
2. forward：model(x, y) でロジットと損失を得る（第10章）
3. backward：loss.backward()（autograd、第3章で体感）
4. step：optimizer.step() でパラメータ更新
```

モデルの中身は複雑になりましたが、外から見た学習の手順は、第3巻のまま。

これが、`nn.Module` と autograd の、ありがたさです。

### 11.2　optimizer（AdamW）と学習率

optimizer は、AdamW を使います。

Transformer の学習の定番で、論文も Adam 系を使っていました（第6巻9章、第2巻11章、第3巻7章）。

```python
import torch

model = SmallGPT(vocab_size=vocab_size, d_model=64, n_heads=4,
                 n_layers=3, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

`lr=3e-4`（= 0.0003）は、小さな Transformer で、よく使われる、手堅い学習率です。

第3巻11章で触れたウォームアップを足すこともできますが、Pre-LN（第9章）なら、なくても安定して学習しやすいです（だから Pre-LN を選んだのでした）。

### 11.3　学習ループを回す

すべてを、つなぎます。

```python
import torch

model.train()                     # 学習モード（Dropout を有効に）
for step in range(3000):
    x, y = get_batch(data, batch_size=16, block_size=block_size)   # 第6章

    logits, loss = model(x, y)    # forward（損失も）
    optimizer.zero_grad()         # 勾配を消す（第4章）
    loss.backward()               # backward（autograd）
    optimizer.step()              # 更新

    if step % 300 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")
```

ループの中身を、確認しましょう。

`get_batch` で、毎ステップ、新しいバッチを取る（第6章）。

`model(x, y)` で、forward して、損失を得る（第10章）。

`zero_grad → backward → step` で、勾配を消して、計算して、更新する。

第3巻7章の骨格、そのままです。

`model.train()` は、Dropout を有効にする、学習モードへの切り替えです（第3巻11章）。

生成・評価のときは、`model.eval()` に切り替えます（次章）。

第2章で素手で書いた一周が、いまや、小さな GPT の学習ループとして、同じ骨格で、動いています。

### 11.4　損失が下がるのを確認する

学習が進めば、損失が、下がっていきます。

```text
step    0  loss 3.91     ← 最初はほぼランダム（log(vocab_size) 付近）
step  300  loss 2.80
step  600  loss 2.31
step  900  loss 2.05
...
```

最初の損失に、注目してください。

`log(vocab_size)` 付近（語彙50なら `log(50) ≈ 3.9`）から、始まっています。

なぜでしょうか。

学習前のモデルは、全文字を、だいたい等確率で予測しています。

「次の文字は、50文字のどれも、同じくらいありそう」という状態です。

このときの損失が、ちょうど `log(vocab_size)` になるのです。

ここから損失が下がっていけば、「次の文字を、だんだん当てられる状態」へ、進んでいる証拠です。

「全文字が等確率」から「特定の文字を高確率で予測」へ、モデルが賢くなっているのです。

### 11.5　よくある詰まり

学習がうまくいかないとき、見るべき定番ポイントです（次章のデバッグ章にも、つながります）。

```text
- 損失が下がらない    : zero_grad の入れ忘れ、学習率が小さすぎる
- 損失が NaN         : 学習率が大きすぎる、mask や softmax の不具合
- 損失が log(V) から動かない : ターゲットのずれ（第6章）が間違っている、出力層の shape ミス
- 出力がおかしい      : 出力層で softmax を二重がけしていないか（第3巻5章）
```

特に、第6章の「ターゲットは入力の1つ後ろ」が、ずれていると、学習が成立しません。

たとえば、ターゲットを1つずらし忘れて、入力と同じにしてしまうと、モデルは「いまの文字をそのまま出す」ことを学んでしまい、意味がありません。

詰まったら、まず `x` と `y` を `print` して、1つずれになっているか、確認するのが、近道です。

「頭で考えるより、まず print」。

第3巻8章でも言った、デバッグの鉄則です。

学習ループができ、損失が下がりました。

あとは、学習したモデルに、文字を生成させるだけです（次章）。

### 11.6　本章のまとめ

- 学習ループの骨格は第3巻7章のまま：get_batch → forward → zero_grad → backward → step。モデルが複雑でも変わらない。
- optimizer は AdamW（lr=3e-4 が手堅い）。Pre-LN なら、ウォームアップなしでも安定しやすい。
- `model.train()` で Dropout を有効に。損失が `log(vocab_size)` 付近（等確率の状態）から下がれば順調。
- 詰まったら zero_grad・学習率・ターゲットのずれ・softmax 二重がけ・shape を疑う。まず print して見る。

---

<!-- chapter-nav:start -->

[← 第10章　モデル全体を組む（小さな GPT）](Chapter%2010%20Assembling%20a%20Small%20GPT.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第12章　文字を生成させる →](Chapter%2012%20Generating%20Text.md)

<!-- chapter-nav:end -->
