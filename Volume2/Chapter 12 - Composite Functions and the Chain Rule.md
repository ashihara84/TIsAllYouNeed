# 第12章 合成関数と連鎖律

**この章のゴール**

ニューラルネットワークを関数の合成として見て、連鎖律とbackpropagationが勾配を前の層へ伝える仕組みだと理解すること。

## 12.1 ニューラルネットワークは関数の合成である

前章の勾配降下法では、各パラメータに対する勾配 `∂loss / ∂parameter`（このパラメータを少し変えたらlossがどう変わるか）が必要でした。しかしニューラルネットワークでは、パラメータからlossまでの距離はかなり遠いです。Transformerでは「token_ids → embedding → Self-Attention → Feed Forward Network → LayerNorm →（何層も）→ output layer → logits → cross entropy → loss」と、たくさんの関数を順番に通った結果としてlossが出ます。

このように関数を順番につなげたものを **合成関数** と呼びます。`h = f(x)`、`y = g(h)` なら、まとめて `y = g(f(x))` です。ニューラルネットワークも基本的にこの合成関数で、各層は入力テンソルを別のテンソルへ変換する関数だと見なせます（embedding: token_idをベクトルに、Self-Attention: 文脈を反映したベクトル列に、output layer: 語彙スコアに、など）。

勾配を計算するには合成関数の微分を理解する必要があり、そこで出てくるのが **連鎖律** です。

---

## 12.2 合成関数とは何か

合成関数とは、ある関数の出力を次の関数の入力にするものです。`f(x) = 2x`、`g(h) = h + 3` なら、`x = 4` のとき `h = f(4) = 8`、`y = g(8) = 11` です。

ニューラルネットワークでも同じことが起きています。小さなネットワークなら次のように関数を順番につなげたものです。

```text
x → Linear → h → ReLU → a → Linear → logits → cross entropy → loss
```

合成関数では途中の値が次の計算に使われるので、最初の方のパラメータがlossにどう影響するかを考えるには途中の計算をたどる必要があります。たとえば `W_Q`（Query用の重み）は直接lossを作っていませんが、変わると次々に影響が伝わります。

```text
W_Qが変わる → Qが変わる → scoresが変わる → weightsが変わる → 出力が変わる → lossが変わる
```

このつながりを通して勾配を計算するために、連鎖律が必要になります。

---

## 12.3 連鎖律とは何か

**連鎖律** とは、合成関数の微分を計算するためのルールです。`h = f(x)`、`y = g(h)`（つまり `y = g(f(x))`）で `dy/dx` を求めたいとき、`x` は直接 `y` に変わるのではなく途中に `h` があります（`x → h → y`）。そこで変化を2段階に分けます。

```text
dy/dx = dy/dh * dh/dx
（xがyに与える影響 = xがhに与える影響 × hがyに与える影響）
```

直感的には「途中の変化を掛け合わせる」ということです。例として `h = 2x`、`y = 3h` を考えます。`dh/dx = 2`、`dy/dh = 3` なので、

```text
dy/dx = 3 * 2 = 6
```

実際 `y = 3h = 3(2x) = 6x` なので、`x` が1増えると `y` は6増えます。このように連鎖律は、関数が何段階にもつながっているときに変化の影響をたどるためのルールです。

---

## 12.4 出力の誤差を前の層へ伝える

ニューラルネットワークの学習では、まず最後にlossが計算されます。しかし更新したいパラメータ（embedding、Q/K/V、Attention出力、FFN、LayerNorm、出力層の重み）はネットワークの中のあちこちにあります。最初の方にあるembeddingやQ/K/Vの重みも更新するには、lossから前の層へ向かって影響をたどる必要があります。

これがバックプロパゲーション（誤差逆伝播）の考え方で、最後に出たlossの情報をネットワークの後ろから前へ伝えていきます。ここで使われる数学的なルールが連鎖律です。たとえば `x → h → logits → loss` で `x` に関する勾配を知りたいなら、次のようにたどります。

```text
∂loss / ∂x = ∂loss / ∂logits × ∂logits / ∂h × ∂h / ∂x
```

これは連鎖律そのものです。人間がすべて手で計算する必要はなく、PyTorchが自動微分で計算してくれます。PyTorchが内部でやっているのは「forwardで計算のつながりを記録 → `loss.backward()` で後ろから前へ勾配を伝える → 各パラメータの `.grad` に勾配を入れる」という流れです。つまり `loss.backward()` は、連鎖律を使ってlossから各パラメータへの勾配を計算していると考えればよいです。

---

## 12.5 backpropagationの数学的な正体

backpropagationの数学的な正体は連鎖律です。ニューラルネットワークは `y = f_n(...f_2(f_1(x)))` のようにたくさんの関数をつなげた合成関数で、最終的にlossを計算します。

ある中間層の重み `W` について `∂loss / ∂W` を知りたいとき、`W` は直接lossにつながっておらず、何段階もの計算を通って影響します。単純な例 `h = xW`、`y = hU`、`loss = L(y)` では、`W → h → y → loss` というつながりをたどります。

```text
∂loss / ∂W = ∂loss / ∂y × ∂y / ∂h × ∂h / ∂W
```

厳密なテンソルの形は少し複雑ですが、直感としてはこれで十分です。バックプロパゲーションは「最後のlossから出発し、各計算を逆向きにたどりながら、連鎖律で勾配を掛け合わせていく」処理です。forwardでは入力から出力へ計算し、backwardではlossから入力側へ勾配を伝えます。この「逆向きにたどる」ことからbackpropagationと呼ばれます。

---

## 12.6 実装では自動微分がやってくれる

連鎖律は重要ですが、実際にTransformerを実装するとき、人間が手で全ての微分を書くことはほとんどありません。PyTorchが自動微分で計算してくれます。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
h = x * 3
y = h ** 2
y.backward()
print(x.grad)
```

この計算は `y = (3x)^2` で、`x = 2` なら `h = 6`、`y = 36` です。微分は `dy/dh = 2h = 12`、`dh/dx = 3` なので、連鎖律より `dy/dx = 12 * 3 = 36`。PyTorchの出力も `tensor(36.)` になります。このようにPyTorchは計算グラフをたどって自動的に連鎖律を適用します。

ニューラルネットワークでも同じで、`loss.backward()` によってモデルの中のすべての学習可能なパラメータについて勾配が計算され、各パラメータの `.grad` に入ります。私たちが手でやるのは「forward計算を書く → lossを計算する → `loss.backward()` を呼ぶ → `optimizer.step()` で更新する」だけです。連鎖律そのものを毎回手で実装する必要はありませんが、勾配がどこからどこへ伝わるのかを理解していないと、モデルがうまく学習しないときに原因を考えにくいので、連鎖律の直感は必要です。

---

## 12.7 それでも連鎖律を知るべき理由

PyTorchが自動微分してくれるなら連鎖律を知らなくてよいのでは、と思うかもしれません。簡単なモデルを動かすだけなら確かに動きます。しかしTransformerを理解して実装できるようになりたいなら、連鎖律の直感は重要です。理由は大きく3つあります。

1つ目は、勾配がどこを通って伝わるかを理解するためです。Transformerではlossから非常に多くの経路を通って勾配が流れ、たとえばQ/K/Vの重みは `W_Q → Q → QK^T → softmax → weights @ V → loss` という流れを通じてlossに影響します。

2つ目は、勾配が消えたり爆発したりする問題を理解するためです。合成関数では微分が何段階にも掛け合わされます。途中で小さい値が何度も掛けられると勾配が非常に小さくなり（`0.1×0.1×0.1×0.1 = 0.0001`、勾配消失）、大きい値が何度も掛けられると非常に大きくなります（`10×10×10×10 = 10000`、勾配爆発）。Transformerで残差接続やLayer Normalizationが重要なのは、深いネットワークでも勾配を流しやすくし学習を安定させるためです。

3つ目は、モデルの構造を設計するときに役立つからです。ニューラルネットワークは勾配によって学習するので、勾配が通らない処理を入れるとその部分はうまく学習できません。たとえばargmaxは離散的に1つを選ぶ操作で通常そのままでは勾配を流しにくく、softmaxは連続的で微分可能なのでAttentionの中で学習に使えます。Attentionでargmaxではなくsoftmaxを使う理由の1つは、重みを滑らかに作れて学習しやすいからです。

---

## 12.8 PyTorchで連鎖律を確認する

単純な合成関数 `h = 3x`、`y = h^2`（つまり `y = (3x)^2`）を確認します。`x = 2` で `h = 6`、`y = 36`、手計算では `dy/dx = (2h)(3) = 12 * 3 = 36` です。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
h = 3 * x
y = h ** 2
y.backward()
print(x.grad)   # tensor(36.)
```

次にニューラルネットワークらしい例として `h = wx + b`、`loss = h^2` を見ます。`x = 2`, `w = 3`, `b = 1` なら `h = 7`、`loss = 49` です。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

h = w * x + b
loss = h ** 2
loss.backward()

print(x.grad, w.grad, b.grad)   # tensor(42.) tensor(28.) tensor(14.)
```

手計算で確認します。`d loss / d h = 2h = 14` なので、`d loss / d w = 14 * x = 14 * 2 = 28`、`d loss / d b = 14 * 1 = 14`、`d loss / d x = 14 * w = 14 * 3 = 42`。PyTorchの出力と一致します。このようにPyTorchは内部で連鎖律を使って各変数への勾配を計算しています。

---

## 12.9 PyTorchの計算グラフ

PyTorchの自動微分は計算グラフ（どの値がどの計算から作られたかを表すつながり）を使って行われます。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

h = x * w
y = h + 1
loss = y ** 2
loss.backward()
print(x.grad, w.grad)
```

PyTorchは `requires_grad=True` のテンソルが関わる計算を記録します。この記録があるから、あとで `loss.backward()` を呼ぶと計算グラフを逆向きにたどり（`loss → y → h → x, w`）、連鎖律で `x` と `w` への勾配を計算できます。

通常の学習ループは「forward計算 → loss計算 → backwardで勾配計算 → optimizerで更新」の順です。

```python
logits = model(inputs)
loss = loss_fn(logits, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`requires_grad=True` が付いていないテンソルだけの計算や、`with torch.no_grad():` の中の計算は、通常は計算グラフに記録されません。たとえばパラメータ更新を手で書くとき、更新操作そのものを計算グラフに含めないために `with torch.no_grad():` の中で行います。学習時のforward計算では勾配が必要ですが、パラメータ更新や推論だけの処理では勾配が不要なことがある、という違いを意識しておくとPyTorchの挙動が理解しやすくなります。

---

## 12.10 小さなネットワークでbackpropagationを見る

1つの線形層を持つ分類モデルでbackpropagationを確認します。

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

print(model.weight.grad)   # ∂loss / ∂weight
print(model.bias.grad)     # ∂loss / ∂bias
```

`loss.backward()` を呼ぶまで勾配は計算されません。呼ぶとモデルのパラメータに勾配が入ります。これらの勾配は、forward（`x → Linear → logits → cross entropy → loss`）を逆向きにたどって計算されています。

```text
loss → cross entropyの勾配 → logitsへの勾配 → Linearへの勾配 → weight, biasへの勾配
```

このように、backpropagationはlossから各パラメータへ勾配を伝える処理です。

---

## 12.11 Transformerにおけるbackpropagationの流れ

Transformerでもbackpropagationの考え方は同じですが、計算の経路がかなり複雑になります。Self-Attentionだけでも次の流れがあります。

```text
x → q=W_Q(x), k=W_K(x), v=W_V(x)
  → scores = q @ k^T / sqrt(d_k)
  → weights = softmax(scores)
  → out = weights @ v
```

この `out` はさらに次の層へ進み、最終的にlossになります。backpropagationではこの流れを逆にたどり、`W_Q`, `W_K`, `W_V` にも勾配が届きます。

これは非常に重要です。Self-Attentionの中のQ/K/Vは手で決めているのではなく、学習によってどのようなQuery・Key・Valueを作ればよいかが調整されます（`W_Q`が更新されるとQueryの作り方が変わる、など）。これにより、モデルはタスクに役立つAttentionのパターンを学習していきます。言語モデルなら、次トークン予測に役立つように各トークンがどのトークンを参照すべきかを学びます。

Transformerの各部品（Linear、行列積、softmax、加算、LayerNorm、活性化関数、cross entropy）は基本的に微分可能な操作でできているので、自動微分で扱えます。だからTransformer全体を1つの大きな合成関数として扱い、lossからすべてのパラメータへ勾配を流せます。学習とは、lossから各パラメータへの勾配を計算して更新することです。

---

## 12.12 detachとno_gradの直感

PyTorchでは `detach()` や `torch.no_grad()` が出てきます。この章の内容と関係するので、直感だけ説明します。

`torch.no_grad()` は、その中の計算を勾配計算の対象にしないためのものです。推論だけなら勾配は不要なので、計算グラフを作らずメモリを節約できます。パラメータ更新を手で行うときにも使い、更新操作自体を学習対象の計算グラフに含めたくないので `no_grad()` の中で行います。

```python
with torch.no_grad():
    w -= learning_rate * w.grad
```

`detach()` は、あるテンソルを計算グラフから切り離します。`y = x.detach()` とすると `y` は `x` と同じ値を持ちますが、そこから先の計算では `x` への勾配が流れません。直感的には「ここで勾配の流れを止める」操作です。通常のTransformer実装では最初から頻繁に使う必要はなく、意味を理解せずに使うと勾配が必要なところで止まって学習できなくなることがあります。この段階では次の理解で十分です。

```text
torch.no_grad(): その範囲の計算を勾配計算の対象にしない
detach(): そのテンソルから過去への勾配の流れを切る
```

どちらも、計算グラフと勾配の流れを制御するための道具です。

---

## 12.13 まとめ

この章では、合成関数と連鎖律について学びました。ニューラルネットワークは多くの関数を順番につなげた合成関数で、Transformerも「token_ids → embedding → Self-Attention → FFN → output layer → logits → cross entropy → loss」という合成関数です。

合成関数の微分を計算するためのルールが連鎖律です。

```text
h = f(x), y = g(h)
dy/dx = dy/dh * dh/dx
（xがhに与える影響 × hがyに与える影響 = xがyに与える影響）
```

ニューラルネットワークでは、lossから各パラメータへの影響を連鎖律によって逆向きにたどります。この処理がbackpropagationです（forward: 入力からlossへ計算 / backward: lossから各パラメータへ勾配を伝える）。

PyTorchでは自動微分（`loss.backward()`）でbackpropagationを行い、各パラメータの `.grad` に勾配が入り、その後 `optimizer.step()` がパラメータを更新します。Transformerでも同じ仕組み（loss → output layer → Transformer blocks → Attention → Q/K/Vの重み → embedding）で学習します。

この章で特に重要なのは、次の理解です。

```text
ニューラルネットワークは合成関数である
連鎖律は合成関数の微分を計算するルールである
backpropagationは連鎖律を使って勾配を逆向きに伝える処理である
PyTorchの自動微分は、このbackpropagationを自動で行ってくれる
```

次章では、正規化について学びます。Transformerでは Layer Normalization が非常に重要です。深いネットワークでは値のスケールが大きすぎたり小さすぎたりすると学習が不安定になります。正規化は、値のスケールを整え、学習を安定させるための重要な道具です。
