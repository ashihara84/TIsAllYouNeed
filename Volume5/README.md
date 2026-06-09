# 第5巻　Transformer の構成要素 — *Heads Are All You Need*

教科書シリーズ「**T Is All You Need**」（全8巻）の第5巻です。
このシリーズの心臓部。Transformer の部品を、**教えるのに最適な順**で一つずつ手で書きます。

## この巻の位置づけ

- **入口（前提）**: 第4巻。
- **出口（読了後にできること）**: Self-Attention・Q/K/V・Multi-Head・位置エンコーディング・FFN を一つずつ理解し、各部品を小さく書いて動かせる。
- **立ち位置**: 第3巻の `LayerNorm(x + Sublayer(x))` の `Sublayer` を、ここで埋める巻。
- **重要**: 部品を作る順番は **論文の順ではない**。易しい核から1つずつ足す、最適な学習順を取る（理由は第1章）。

## この巻でやらないこと（＝次の巻の仕事）

- 論文を頭から精読する → **第6巻**（部品を全部知ってから読む）
- 部品を統合して文字レベル言語モデルを通す → **第7巻**
- Decoder-only 化・事前学習・RLHF など現代 LLM の話 → **第8巻**

各部品を小さく書いて動かすまで。最後の章でも部品を「並べてみる」だけで、言語モデルとしての統合はしない。

## まず読むもの

- [リンク付き目次](Table%20of%20Contents.md): 全章と節へ移動するための目次
- [用語集](Glossary.md): この巻で出てくる基本語の一覧

## 読み方

1. [第1章 この巻の地図と教え方の順番](Chapter%201%20Map%20and%20Teaching%20Order.md) から順番に読む
2. 第2章〜第7章で、Attention の核から Multi-Head まで一段ずつ作る
3. 第8章〜第10章で、位置エンコーディングと FFN を足す
4. [第11章 残差接続と LayerNorm で部品を包む](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md) で Transformer ブロックを完成させる
5. 各章のコードを手元で動かし、shape を `print` で追う

## 構成（全13章）

- 第1章: この巻の地図と教え方の順番
- 第2章〜第7章: 重み付き和 → Q/K/V → スケール付き内積 → Self-Attention → causal mask → Multi-Head
- 第8章〜第10章: 位置エンコーディング（概念・実装）、Feed-Forward Network
- 第11章〜第12章: 残差＋LayerNorm で包む、部品を並べる（統合はしない）
- 第13章: まとめ（部品表、第6巻・第7巻への橋）

## シリーズの中での位置

第1巻 機械学習の基礎 → 第2巻 数学の基礎 → 第3巻 ニューラルネットの基礎 → 第4巻 言語をベクトルにする → 第5巻（本書） → 第6巻 『Attention Is All You Need』精読 → 第7巻 PyTorch で実装 → 第8巻 現代の LLM へ

シリーズ全体の地図は、リポジトリ直上の [Series Table of Contents.md](../Series%20Table%20of%20Contents.md) を参照してください。
