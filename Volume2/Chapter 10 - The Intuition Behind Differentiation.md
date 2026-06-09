# 第10章 微分の直感

**この章のゴール**

微分と勾配を「パラメータを少し変えたときlossがどう変わるかを見る道具」として理解すること。

## 10.1 微分とは何か

この章では **微分** について学びます。ニューラルネットワークでは、損失を小さくするために、パラメータをどちら向きにどれくらい動かせばよいかを知る必要があり、そのために微分が必要です。

モデルにある重み `w` を少し変えると、損失 `loss` も変わります。微分は、ざっくり言えば「ある値を少し変えたとき、結果がどれくらい変わるかを見るもの」です。

たとえば `y = x^2` では、`x` を変えると `y` も変わります（`x=1→y=1`, `x=2→y=4`, `x=3→y=9`）。`x` を少し増やしたら `y` がどれくらい増えるか、それが微分の考え方です。

ニューラルネットワークでは「重みを少し変えたら、lossはどう変わるか」を考えます。重みを増やすとlossが増えるならその重みは減らした方がよく、増やすとlossが減るなら増やした方がよい。微分は「lossを小さくするために、パラメータをどう動かすか」を知るために使われます。

---

## 10.2 変化量を見るという考え方

微分を理解するには、まず「変化量」を考えるとよいです。`y = 2x` では `x` が1増えると `y` はいつでも2増えます。つまり「xの変化量に対して、yは2倍変化する」と言えます。

一方 `y = x^2` では変化量が一定ではありません。

```text
x: 1 → 2 で y: 1 → 4（変化量 3）
x: 2 → 3 で y: 4 → 9（変化量 5）
x: 3 → 4 で y: 9 → 16（変化量 7）
```

`x` が大きくなるほど `y` の増え方も大きくなります。微分は、この「その場所での変化の勢い」を見るための道具です。

ニューラルネットワークでも同じで、あるパラメータを少し変えたときlossがどれくらい変わるかは、今のパラメータの値によって変わります（少し増やすとlossが大きく減る場所、ほとんど変わらない場所、増えてしまう場所がある）。

---

## 10.3 傾きとは何か

微分はよく「傾き」として説明されます。`y = 2x` は直線で、傾きは2です。

```text
傾き = yの変化量 / xの変化量 = 2 / 1 = 2
```

傾きは「入力を少し変えたとき、出力がどれくらい変わるか」を表します。`y = x^2` は曲線で、場所によって傾きが変わります（小さい場所はゆるやか、大きい場所は急）。`y = x^2` の微分は次の通りです。

```text
dy/dx = 2x
（x=1で傾き2, x=2で傾き4, x=3で傾き6）
```

ニューラルネットワークでは `y` の代わりに `loss` を考えます。`loss = f(w)` のとき、`d loss / d w` は重み `w` に対するlossの傾きで、「wを少し変えたとき、lossがどれくらい変わるか」を表します。この傾きがわかると、lossを減らす方向がわかります。

---

## 10.4 損失を小さくする方向を知る

機械学習では損失を小さくしたい（＝モデルの予測をよくする）ので、パラメータをどう動かせばlossが小さくなるかを知る必要があります。ここで微分が役立ちます。

```text
d loss / d w > 0（傾きがプラス）
→ wを増やすとlossが増える → wを減らす

d loss / d w < 0（傾きがマイナス）
→ wを増やすとlossが減る → wを増やす
```

まとめると、lossを小さくするには傾きと逆向きに動かします。これが勾配降下法の基本で、直感は「lossが増える方向がわかる → その逆に動く → lossが下がる」と非常に単純です。微分はこの「lossが増える方向」を教えてくれます。だからニューラルネットワークの学習には微分が必要なのです。

---

## 10.5 パラメータを少し変えると損失はどう変わるか

ニューラルネットワークには大量のパラメータがあります（Transformerなら線形層の重み、バイアス、embedding、LayerNormのパラメータなど）。学習では各パラメータについて「少し増やすとlossは増えるのか減るのか」を知りたいです。

```text
∂loss / ∂w1 = 0.8  → w1を増やすとlossが増える → 減らす
∂loss / ∂w2 = -0.3 → w2を増やすとlossが減る → 増やす
∂loss / ∂w3 = 0.0  → w3を変えてもlossはほぼ変わらない
```

パラメータの数が非常に多いので、人間が手で計算することはできません。そこでPyTorchなどのフレームワークが自動微分で計算してくれます。ただし、何を計算しているのか（各パラメータを少し変えたらlossがどう変わるか）を理解しておくことは重要です。この情報を使ってoptimizerがパラメータを更新します。

---

## 10.6 偏微分とは何か

ここまでは入力が1つの関数（`y = x^2`）を考えてきました。しかしニューラルネットワークではパラメータがたくさんあり、`loss = f(w1, w2, w3)` のようになります。

複数の変数があるとき、1つの変数だけに注目して微分することを **偏微分** と呼びます。たとえば `∂loss / ∂w1` は「w2やw3は固定したまま、w1だけを少し変えたらlossはどう変わるか」を表します。ニューラルネットワークでは各パラメータについて偏微分を計算します。

記号は、1変数の微分では `d y / d x`、複数変数の偏微分では `∂` を使い `∂loss / ∂w` と書きます。読み方は「lossをwで偏微分したもの」ですが、まずは「wを少し変えたらlossがどう変わるか」と読めば十分です。

---

## 10.7 勾配とは何か

**勾配** とは、すべての偏微分をまとめたものです。`loss = f(w1, w2, w3)` で各偏微分が次のようなら、

```text
∂loss/∂w1 = 0.8, ∂loss/∂w2 = -0.3, ∂loss/∂w3 = 0.0
gradient = [0.8, -0.3, 0.0]
```

勾配は、lossが最も増えやすい方向を表します。学習でやりたいのはlossを減らすことなので、勾配とは逆方向に動かします。これが勾配降下法です。

```text
w = w - learning_rate * gradient
```

`learning_rate`（学習率）は、どれくらいの大きさで動かすかを決める値です。たとえば `w = [1.0, 2.0, 3.0]`、`gradient = [0.8, -0.3, 0.0]`、`learning_rate = 0.1` なら、

```text
w_new = [1.0, 2.0, 3.0] - 0.1 * [0.8, -0.3, 0.0] = [0.92, 2.03, 3.0]
```

`w1` は減り（勾配がプラス）、`w2` は増え（勾配がマイナス）、`w3` は変わりません（勾配が0）。このように勾配は、各パラメータをどちら向きに動かすべきかを教えてくれます。

---

## 10.8 PyTorchで微分を確認する

`y = x^2` の微分は `dy/dx = 2x` なので、`x = 3` での傾きは `2 * 3 = 6` です。PyTorchで確認します。

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()

print("y:", y)            # tensor(9., grad_fn=<PowBackward0>)
print("x.grad:", x.grad)  # tensor(6.)
```

`x.grad` が `6` になっています。重要なのは `requires_grad=True` で、これはPyTorchに「この値について勾配を計算したい」と伝える指定です。`y.backward()` を呼ぶとPyTorchが自動微分で勾配を計算し、結果が `x.grad` に入ります。

---

## 10.9 PyTorchでlossの勾配を確認する

機械学習らしい例を見ます。単純なモデル `y_pred = w * x` で、入力 `x = 2.0`、正解 `y_true = 10.0`、`w = 3.0` なら予測は `6.0` で、正解10より小さすぎます。損失を二乗誤差 `loss = (y_pred - y_true)^2` で定義します。

```python
import torch

x = torch.tensor(2.0)
y_true = torch.tensor(10.0)
w = torch.tensor(3.0, requires_grad=True)

y_pred = w * x
loss = (y_pred - y_true) ** 2
loss.backward()

print("loss:", loss)      # tensor(16.)
print("w.grad:", w.grad)  # tensor(-16.)
```

`w.grad` が `-16`（マイナス）なので「wを増やすとlossが減る」という意味です。勾配降下法で `learning_rate = 0.1` で更新すると、

```text
w_new = 3.0 - 0.1 * (-16) = 3.0 + 1.6 = 4.6
```

`w` は増えます。今の予測は6で正解は10なので、予測を大きくするために `w` を増やすのは直感と合っています。

```python
learning_rate = 0.1
with torch.no_grad():
    w -= learning_rate * w.grad
print("updated w:", w)   # tensor(4.6000, requires_grad=True)
```

---

## 10.10 勾配を使った1ステップの更新

PyTorchで1ステップの学習をまとめて書きます。やることは「予測 → loss計算 → backwardで勾配計算 → 勾配で更新 → 勾配をリセット」です。

```python
import torch

x = torch.tensor(2.0)
y_true = torch.tensor(10.0)
w = torch.tensor(3.0, requires_grad=True)
learning_rate = 0.1

y_pred = w * x
loss = (y_pred - y_true) ** 2
loss.backward()

with torch.no_grad():
    w -= learning_rate * w.grad
w.grad.zero_()

print("updated w:", w)
```

`with torch.no_grad():` は、この更新操作自体を勾配計算の対象にしないためのものです。最後の `w.grad.zero_()` が重要で、PyTorchでは勾配が上書きではなく加算されるので、次のステップに進む前にゼロにする必要があります。

学習では「予測 → loss → backward → update → gradをzero → 次のデータへ」を何度も繰り返します。実際は手で `w -= ...` と書く代わりにoptimizer（`optimizer.step()`, `optimizer.zero_grad()`）を使うことが多いですが、最初は手で更新してみると微分と勾配降下法の意味がよくわかります。

---

## 10.11 Transformerで微分はどこに関係するのか

Transformerでは多くの計算（embedding、Q/K/Vの線形変換、QK^T、softmax、weights @ V、Feed Forward Network、LayerNorm、出力層、cross entropy loss）が組み合わさって、最終的にlossが出ます。

学習ではこのlossを小さくしたいので、Transformerの中のすべての学習可能なパラメータ（embedding table、W_Q、W_K、W_V、Attention output projection、FFNの重み、LayerNormのパラメータ、出力層の重み）について勾配 `∂loss / ∂parameter` を計算します。

Transformerの中の計算は多段階です。たとえば `W_Q` は直接lossを出しているのではなく、まずQueryを作り、それがAttention scoreに使われ、softmaxに入り、Valueを混ぜ、次の層に渡り、最終的にlogitsになり、lossになります。

```text
W_Q → Q → Attention → 次の層 → logits → loss
```

このように遠く離れたパラメータが最終的なlossに影響します。この影響を連鎖的に計算する仕組みがバックプロパゲーションです（後の章で扱います）。今は次のことを理解しておけば十分です。

```text
Transformerの学習では、lossから各パラメータへの勾配を計算する
PyTorchは自動微分でそれを計算してくれる
勾配を使って、lossが小さくなる方向にパラメータを更新する
```

---

## 10.12 PyTorchの自動微分とニューラルネットワーク

小さなニューラルネットワークで自動微分を確認します。入力ベクトルを線形層に通し、lossを計算し、`backward()` で各パラメータの勾配を求めます。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([1])

model = nn.Linear(3, 2)
logits = model(x)
loss = F.cross_entropy(logits, target)
loss.backward()

print(model.weight.grad)   # 重みに対する勾配
print(model.bias.grad)     # バイアスに対する勾配
```

流れは「x → Linear(3,2) → logits → cross entropy → loss → backward → 勾配」です。`model.weight.grad` には「重みを少し変えたらlossがどう変わるか」が入ります。

実際の学習ではoptimizerを使います。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([1])

model = nn.Linear(3, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

logits = model(x)
loss = F.cross_entropy(logits, target)

optimizer.zero_grad()   # 前回の勾配をリセット
loss.backward()         # 今のlossに対する勾配を計算
optimizer.step()        # 勾配を使ってパラメータを更新
```

この3つ（`zero_grad`, `backward`, `step`）がニューラルネットワークの学習ループの基本です。実際はこれをミニバッチごとに繰り返します。

```python
for batch in data:
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Transformerの学習でも基本構造は同じで、モデルが大きくなってもやっていること（予測する・lossを計算する・勾配を計算する・パラメータを更新する）は変わりません。

---

## 10.13 まとめ

この章では、微分の直感について学びました。微分は、ある値を少し変えたときに結果がどれくらい変わるかを見る道具です。ニューラルネットワークでは「パラメータを少し変える → lossがどれくらい変わるか」を見ます。この情報がわかると、lossを小さくする方向がわかります（増やすとlossが増えるなら減らす、減るなら増やす）。

複数のパラメータがある場合、各パラメータについて微分を考えます。これを偏微分と呼び、すべての偏微分をまとめたものが勾配です。

```text
gradient = [∂loss/∂w1, ∂loss/∂w2, ∂loss/∂w3, ...]
```

勾配はlossが最も増えやすい方向を表すので、lossを小さくするにはその逆向きに動きます。

```text
w = w - learning_rate * gradient
```

PyTorchでは自動微分（`loss.backward()`）で勾配を計算でき、各パラメータの `.grad` に入ります。実際の学習ではoptimizerを使います。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

この章で特に重要なのは、次の理解です。

```text
微分は「少し変えたときの変化」を見る
勾配は各パラメータに対するlossの変化をまとめたもの
勾配の逆方向に動くとlossが下がりやすい
PyTorchは自動微分で勾配を計算してくれる
```

次章では、勾配降下法について学びます。この章では「勾配とは何か」を見ました。次章では、その勾配を使って実際にどのようにパラメータを更新し、モデルを学習させるのかを詳しく見ていきます。

---

<!-- chapter-nav:start -->

[← 第9章 損失関数とクロスエントロピー](Chapter%209%20-%20Loss%20Functions%20and%20Cross-Entropy.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第11章 勾配降下法 →](Chapter%2011%20-%20Gradient%20Descent.md)

<!-- chapter-nav:end -->
