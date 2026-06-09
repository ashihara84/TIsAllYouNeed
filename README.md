# T Is All You Need

Transformer を **自分で（PyTorch で）実装できるようになる**ための、日本語の教科書シリーズ（全8巻）です。機械学習の基礎から始めて、数学・ニューラルネット・言語のベクトル化・Attention の構成要素・論文精読・実装・現代の LLM までを、一本の階段としてつなぎます。

各巻は `VolumeN/` フォルダにあり、章ごとの本文（`.md`）・`README`・`Glossary`（用語集）・リンク付きの `Table of Contents.md`（目次）を備えています。シリーズ全体の目次は [Series Table of Contents.md](Series%20Table%20of%20Contents.md) にあります。

## 各巻

| 巻 | テーマ | 副題 | 実装の度合い |
|----|--------|------|------|
| [第1巻](Volume1/Table%20of%20Contents.md) | 機械学習の基礎（全14章） | *Learning Is All You Need* | なし（直感） |
| [第2巻](Volume2/Table%20of%20Contents.md) | 数学の基礎（全16章） | *Math Is All You Need* | コードで一度確かめる |
| [第3巻](Volume3/Table%20of%20Contents.md) | ニューラルネットの基礎（全12章） | *Layers Are All You Need* | 小さい MLP を書く |
| [第4巻](Volume4/Table%20of%20Contents.md) | 言語をベクトルにする／系列（全12章） | *Tokens Are All You Need* | 動機づけ（RNN は実装しない） |
| [第5巻](Volume5/Table%20of%20Contents.md) | Transformer の構成要素（全13章） | *Heads Are All You Need* | 各部品を小さく書く |
| [第6巻](Volume6/Table%20of%20Contents.md) | 『Attention Is All You Need』精読（全11章） | **Attention Is All You Need** | 薄く鋭く・実装なし |
| [第7巻](Volume7/Table%20of%20Contents.md) | PyTorch で Transformer を実装（全14章） | *Code Is All You Need* | 素手 → PyTorch で統合 |
| [第8巻](Volume8/Table%20of%20Contents.md) | 現代の LLM へ（全12章） | *Scale Is All You Need* | 回収と位置づけ |

## このシリーズのねらい

最終的に次の式を読み、PyTorch のコードに落とせるようになることを目指します。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

そのために、数式を「意味・shape・実装の対応」の3点セットで読む姿勢を一貫させています。

## 読む経路（読者タイプ別の入口）

- **機械学習が未経験のエンジニア**：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8（全部）
- **機械学習は既習（数学に不安）**：2 から（1 は飛ばす）
- **機械学習・数学とも既習**：3 から（1・2 を飛ばす）
- **論文輪読まで済んでいる人**：3 → 4 → 5 で部品を作り、6 で論文を回収、7 で実装
- **PyTorch 実装だけ読みたい人**：7 へ（素手パートは対比のため冒頭だけ眺める）
