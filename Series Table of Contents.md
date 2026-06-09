# T Is All You Need — シリーズ目次（全8巻）

Transformer を自分で（PyTorch で）実装できるようになるための教科書シリーズ。
各巻は `VolumeN/` フォルダにあり、本文（章ごとの `.md`）・README・用語集・リンク付き目次（`Table of Contents.md`）を備える。

## 各巻へのリンク

- 第1巻 [機械学習の基礎](Volume1/Table%20of%20Contents.md)（全14章）
- 第2巻 [数学の基礎](Volume2/Table%20of%20Contents.md)（全16章）
- 第3巻 [ニューラルネットの基礎](Volume3/Table%20of%20Contents.md)（全12章）
- 第4巻 [言語をベクトルにする／系列](Volume4/Table%20of%20Contents.md)（全12章）
- 第5巻 [Transformer の構成要素](Volume5/Table%20of%20Contents.md)（全13章）
- 第6巻 [『Attention Is All You Need』精読](Volume6/Table%20of%20Contents.md)（全11章）
- 第7巻 [PyTorch で Transformer を実装](Volume7/Table%20of%20Contents.md)（全14章）
- 第8巻 [現代の LLM へ](Volume8/Table%20of%20Contents.md)（全12章）

## 読む経路図（読者タイプ別の入口）

- **ML 未経験のエンジニア**：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8（全部）
- **ML 既習（数学に不安）**：2 から（1 は飛ばす）
- **ML・数学とも既習**：3 から（1・2 を飛ばす）
- **論文輪読済みのゼミ生**：3 → 4 → 5 で部品を作り、6 で論文を回収、7 で実装
- **PyTorch だけ読みたい人**：7 へ。素手パート（7巻2章）は対比のため冒頭だけ眺める

## 各巻の入口と出口

1. **機械学習の基礎** — 入口: なし / 出口: 教師あり学習・損失・勾配降下を言葉で説明できる（直感どまり）
2. **数学の基礎** — 入口: 1巻 / 出口: ベクトル・行列・内積・softmax・微分が記号で怖くない（コードで一度確認）
3. **ニューラルネットの基礎** — 入口: 1・2巻 / 出口: 小さい MLP を書ける。残差接続・LayerNorm を仕込む
4. **言語をベクトルにする／系列** — 入口: 3巻 / 出口: tokenization・embedding・言語モデルが分かり、RNN の辛さと Attention が足したものを語れる
5. **Transformer の構成要素** — 入口: 4巻 / 出口: Self-Attention・Q/K/V・Multi-Head・位置エンコーディング・FFN を各部品ごとに書いて動かす
6. **『Attention Is All You Need』精読** — 入口: 5巻 / 出口: 論文の順で読み、提示の仕方と設計理由だけを見る（薄く鋭い）
7. **PyTorch で実装** — 入口: 6巻 / 出口: 部品を統合し、文字レベル言語モデルが次の文字を吐くまで通す
8. **現代の LLM へ** — 入口: 7巻 / 出口: いま触る GPT がこの全体のどこを削いだ形かを回収
