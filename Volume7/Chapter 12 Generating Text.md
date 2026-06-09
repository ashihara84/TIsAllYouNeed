# 第12章　文字を生成させる

**この章でわかること**

- 次の文字を、確率分布から選ぶ
- temperature とサンプリング（第2巻8章の回収）
- 文章を1文字ずつ、伸ばす
- 学習前後で、出力を比べる

この章が、シリーズ全体のクライマックスです。

学習した小さな GPT に、実際に文字を、吐かせます。

第1巻から続いた旅の、到達点です。

### 12.1　次の文字を確率分布から選ぶ

生成の仕組みは、第4巻7章で学んだとおりです。

「これまでの文字列をモデルに入れ、次の文字の分布を得て、1つ選び、末尾に足す」を、繰り返します。

モデルの出力は、各位置のロジット `[B, L, vocab_size]` でした。

生成では、**いちばん最後の位置** のロジットだけを、使います。

なぜでしょうか。

それが「次の文字」の予測だからです。

「犬が走」と入れたら、最後の「走」の位置のロジットが、「次に来る文字」を表しています。

```text
これまで: "犬が走"
→ モデルに入れる
→ 最後の位置のロジット → softmax → 次の文字の確率分布
→ 1つ選ぶ（例: "る"）
→ "犬が走る" にして、また繰り返す
```

### 12.2　temperature とサンプリング（第2巻8章の回収）

次の文字を、どう選ぶか。

第2巻8章で学んだ、temperature とサンプリングを、使います。

**temperature。**

ロジットを `temperature` で割ってから、softmax します。

temperature が小さいと、分布がとがり、手堅い（でも繰り返しがちな）選択になります。

temperature が大きいと、分布が平らになり、多様（でも乱れがちな）選択になります。

```text
temperature 低い : 安全だが単調（同じような文字ばかり）
temperature 高い : 多様だが崩れやすい（変な文字も出る）
（1.0 付近が標準的な出発点）
```

**サンプリング。**

得た分布から、`torch.multinomial` で、サンプリングします。

いつも最大を選ぶ（貪欲）より、確率に従ってランダムに選ぶほうが、自然で多様な文に、なります。

### 12.3　文章を1文字ずつ伸ばす

生成を、`nn.Module` のメソッドとして、書きます。

推論なので `torch.no_grad()` で囲み、`model.eval()` で Dropout を切ります（第4章・第11章）。

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        # block_size を超えないよう、末尾だけを使う
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)            # [B, L, vocab_size]
        logits = logits[:, -1, :] / temperature  # 最後の位置だけ [B, vocab_size]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # サンプリング [B, 1]
        idx = torch.cat([idx, next_id], dim=1)              # 末尾に足す
    return idx
```

一行ずつ、追いましょう。

`idx[:, -model.block_size:]` は、文脈が `block_size` を超えたら、古い部分を捨てる処理です。

モデルは `block_size` までしか位置を持たない（第7章の `max_len`）ので、それを超えないようにします。

`logits[:, -1, :]` で、最後の位置のロジットだけを、取り出します。

`/ temperature` で、温度をかけます。

`F.softmax` で、確率分布にします。

`torch.multinomial` で、その分布から、次の文字を1つ、サンプリングします。

`torch.cat` で、選んだ文字を、系列の末尾に足します。

これを繰り返します。

第4巻7章の生成ループ図が、そのまま、コードになっています。

### 12.4　学習前後で出力を比べる

生成を、実行します。

種となる文字（たとえば最初の1文字）から始めて、続きを生成させます。

```python
import torch

# 種：最初の文字の ID（例として 0）
start = torch.zeros((1, 1), dtype=torch.long)

out_ids = generate(model, start, max_new_tokens=100, temperature=1.0)
print(decode(out_ids[0].tolist()))   # 第6章の decode で文字列に戻す
```

学習 **前** のモデルだと、出力は、デタラメな文字の羅列です。

まだ何も学んでいないので、当然です。

学習 **後** のモデルだと、学習テキストの癖を反映した、それらしい文字の並びが、出てきます。

```text
学習前: "ﾞ｜３｝あ７…"（デタラメ）
学習後: "犬が走る。猫も歩く。"（学習データの癖を反映した並び）
```

小さなモデル・小さなデータなので、完璧な文には、なりません。

でも、「次の文字をモデルが予測して、つないでいる」ことは、はっきり見て取れます。

学習前のデタラメと、学習後のそれらしさ。

この差が、「学習した」ことの、証拠です。

この瞬間――**自分で組んだ Transformer が、自分で文字を生成した**――が、第1巻から続いた旅の、到達点です。

embedding も、Attention も、残差も、LayerNorm も、学習ループも、生成も、すべて、自分の手を通ってきました。

ブラックボックスだった「次の文字を予測する仕組み」が、いまや、隅々まで自分のものに、なったのです。

### 12.5　本章のまとめ

- 生成は「最後の位置のロジット → softmax → サンプリング → 末尾に足す」の繰り返し（第4巻7章）。
- temperature でロジットを割って分布の鋭さを変え、`multinomial` でサンプリングする（第2巻8章）。
- 生成は `torch.no_grad()` ＋ `model.eval()`。文脈は `block_size` までに切り詰める。
- 学習後は、学習データの癖を反映した、それらしい文字列が出る。自分で組んだ Transformer が、文字を生成した。

---

<!-- chapter-nav:start -->

[← 第11章　学習ループを書く](Chapter%2011%20Writing%20the%20Training%20Loop.md) ｜ [目次](Table%20of%20Contents.md) ｜ [第13章　デバッグと shape 地獄の歩き方 →](Chapter%2013%20Debugging%20and%20Shape%20Hell.md)

<!-- chapter-nav:end -->
