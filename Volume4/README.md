# 第4巻　言語をベクトルにする／系列を扱う — *Tokens Are All You Need*

教科書シリーズ「**T Is All You Need**」（全8巻）の第4巻です。
言語をニューラルネットで扱うための土台を組み立て、最後に「なぜ Attention が必要になったのか」を腑に落とします。Transformer 前夜の巻です。

## この巻の位置づけ

- **入口（前提）**: 第3巻。
- **出口（読了後にできること）**: tokenization・embedding・言語モデル（次トークン予測）が分かり、「RNN は何が辛くて、Attention が何を足したか」を自分の言葉で語れる。
- **立ち位置**: Attention が必要になる **動機** を作る巻。

## この巻でやらないこと（＝次の巻の仕事）

- RNN / LSTM を **実装** する（この巻では動機づけの比較対象に留め、実装しない）
- Self-Attention・Q/K/V を実装する → **第5巻**
- Transformer を組む → **第7巻**
- 現代の LLM のトークナイザ（BPE など）の詳細 → **第8巻**（この巻では概念まで）

## まず読むもの

- [リンク付き目次](Table%20of%20Contents.md): 全章と節へ移動するための目次
- [用語集](Glossary.md): この巻で出てくる基本語の一覧

## 読み方

1. [第1章 この巻の地図](Chapter%201%20Map%20of%20This%20Volume.md) から順番に読む
2. 前半（第2〜7章）で、言語をベクトルにし、系列と言語モデルを理解する
3. 後半（第8〜11章）で、n-gram → RNN → Attention の流れを追い、Attention の動機をつかむ
4. [第11章 Attention は何を足したか](Chapter%2011%20What%20Attention%20Added.md) が、この巻のクライマックス

## 構成（全12章）

- 第1章: この巻の地図
- 第2章〜第5章: 言語を数にする、トークン化、語彙と ID、embedding
- 第6章〜第7章: 系列と文脈、言語モデル（次トークン予測）
- 第8章〜第10章: n-gram からニューラル言語モデルへ、RNN の考え方、RNN の辛さ
- 第11章: Attention は何を足したか（動機づけ）
- 第12章: まとめ（第5巻への橋）

## シリーズの中での位置

第1巻 機械学習の基礎 → 第2巻 数学の基礎 → 第3巻 ニューラルネットの基礎 → 第4巻（本書） → 第5巻 Transformer の構成要素 → 第6巻 『Attention Is All You Need』精読 → 第7巻 PyTorch で実装 → 第8巻 現代の LLM へ

シリーズ全体の地図は、リポジトリ直上の [Series Table of Contents.md](../Series%20Table%20of%20Contents.md) を参照してください。
