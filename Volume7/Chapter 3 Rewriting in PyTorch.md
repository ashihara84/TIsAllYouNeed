# 第3章　同じネットを PyTorch で書き直す

**この章でわかること**

- 前章の素手のネットを、テンソルに置き換える
- `loss.backward()` 一行で、逆伝播が消える
- 素手版と数値を突き合わせる
- 対比から得られる、納得

### 3.1　テンソルに置き換える

前章で NumPy で書いた XOR の MLP を、PyTorch で、書き直します。

まずは `nn.Module` を使わず、前章と、できるだけ同じ形で、テンソルとして書いてみます。

こうすると、何が変わって、何が変わらないかが、はっきりします。

```python
import torch

torch.manual_seed(0)
X = torch.tensor([[0.,0.],[1.,1.],[0.,1.],[1.,0.]])
y = torch.tensor([0, 0, 1, 1])

# パラメータ：requires_grad=True で「勾配を計算してほしい」と印をつける
W1 = (torch.randn(2, 8) * 0.5).requires_grad_(True)
b1 = torch.zeros(8, requires_grad=True)
W2 = (torch.randn(8, 2) * 0.5).requires_grad_(True)
b2 = torch.zeros(2, requires_grad=True)
```

前章の NumPy 版と、見比べてください。

`np.array` が `torch.tensor` に、`np.random.randn` が `torch.randn` に、変わっただけです。

大きく違うのは、一点だけ。

`requires_grad=True` という印です。

これは、第2巻10章・第3巻6章で学んだ印でした。

「このパラメータについて、勾配を計算してほしい」という印です。

これをつけたテンソルについて、PyTorch は計算を記録し、勾配を自動で求めてくれます。

### 3.2　`loss.backward()` 一行で逆伝播が消える

順伝播と損失は、前章と、ほぼ同じ式です。

ところが、逆伝播が、劇的に変わります。

前章では、`backward()` 関数を、何十行も手で書きました。

`do`、`dW2`、`dh`、`dz1`、`dW1` ... と、一つずつ。

今回は、その全部が、**一行** になります。

```python
loss_fn = torch.nn.CrossEntropyLoss()

lr = 0.5
for step in range(2000):
    # 順伝播（前章と同じ式）
    z1 = X @ W1 + b1
    h = torch.relu(z1)
    o = h @ W2 + b2
    loss = loss_fn(o, y)        # softmax + 交差エントロピー（内部で）

    # 逆伝播：前章の backward() 関数 全体が、この一行に化ける
    loss.backward()

    # 更新（勾配を使って手で更新。autograd の外で行うので no_grad）
    with torch.no_grad():
        for p in (W1, b1, W2, b2):
            p -= lr * p.grad
            p.grad.zero_()        # 勾配を消す（溜まるので）

    if step % 400 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")
```

注目すべきは、`loss.backward()` の、たった一行です。

前章の `do`、`dW2`、`db2`、`dh`、`dz1`、`dW1`、`db1` を導く計算。

softmax の勾配、線形層を逆向きに通す `do @ W2.T`、ReLU の `dh * (z1 > 0)`。

その **すべて** が、`loss.backward()` の一行に、置き換わりました。

`p.grad` に、手で計算したのと同じ勾配が、自動的に入っています。

前章で何十行も書いたものが、一行になる。

これが、この巻でいちばん味わってほしい、瞬間です。

なお、更新の部分を `with torch.no_grad()` で囲んでいるのは、「パラメータを更新する計算自体は、勾配を追跡しなくてよい」からです（第2巻12章）。

`p.grad.zero_()` で、毎ステップ勾配を消すのも、前章と同じ（勾配は溜まるので）です。

### 3.3　素手版と数値を突き合わせる

「本当に、同じ勾配が出ているのか？」

これを確かめると、対比が、いっそう腑に落ちます。

同じ初期値・同じ入力で、前章の手書き `backward` の結果と、PyTorch の `p.grad` を、比べると、一致します（数値誤差の範囲で）。

```python
# 概念の確認：手書きの dW2 と、PyTorch の W2.grad は一致する
# （同じ式を計算しているので当然だが、確かめると納得が深まる）
```

つまり、PyTorch は、魔法では、ありません。

前章であなたが手で書いた、連鎖律の計算を、計算グラフをたどって、自動で実行しているだけです。

「自動で」やってくれるけれど、やっていることは、あなたが手で書いたのと、同じ。

中身を知っているからこそ、`backward()` を、信頼して使えます。

「よく分からないけど、便利だから使う」のと、「中身を知ったうえで、便利だから使う」のとでは、安心感がまるで違います。

### 3.4　対比から得られる納得

この2章の対比で、得られたものを、整理します。

```text
素手（第2章）           PyTorch（本章）
─────────────────────────────────────
forward を手で書く   →  ほぼ同じ式（テンソルで）
損失を手で書く        →  CrossEntropyLoss
backward を手で書く   →  loss.backward() の一行 ★
更新を手で書く        →  p -= lr * p.grad（or optimizer）
```

★の行が、autograd の、ありがたみです。

逆伝播は、層が増え、Attention のように複雑になるほど、手で書くのが、現実的でなくなります。

だから第1章で「Attention まで素手で行かない」と、決めたのでした。

その重労働を、肩代わりしてくれるのが、autograd です。

ここから先、この巻では、ずっと PyTorch を使います。

`loss.backward()` を、安心して使えるのは、その一行の中身を、第2章で一度、自分の手で触ったからです。

次章から、PyTorch の道具（テンソル・autograd・`nn.Module`）を整理し、いよいよ小さな GPT の組み立てに、入ります。

### 3.5　本章のまとめ

- 前章の素手 MLP を PyTorch で書き直すと、手書きの `backward()` 全体が、`loss.backward()` 一行になる。
- `requires_grad=True` のテンソルについて、PyTorch が計算を記録し、`p.grad` に勾配を入れてくれる。
- 手書きの勾配と PyTorch の勾配は、一致する。autograd は魔法でなく、連鎖律の自動実行。
- 中身を一度手で触ったからこそ、`backward()` を信頼して使える。以降はずっと PyTorch。
