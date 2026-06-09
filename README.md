# T Is All You Need

Transformer を **自分で（PyTorch で）実装できるようになる**ための、日本語の教科書シリーズ（全8巻）です。機械学習の基礎から始めて、数学・ニューラルネット・言語のベクトル化・Attention の構成要素・論文精読・実装・現代の LLM までを、一本の階段としてつなぎます。

各巻は `VolumeN/` フォルダにあり、章ごとの本文（`.md`）・`README`・`Glossary`（用語集）・リンク付きの `Table of Contents.md`（目次）を備えています。シリーズ全体の目次は [Series Table of Contents.md](Series%20Table%20of%20Contents.md) にあります。

## 各巻

| 巻 | テーマ | 副題 |
|----|--------|------|
| [第1巻](Volume1/README.md) | 機械学習の基礎（全14章） | *Learning Is All You Need* |
| [第2巻](Volume2/README.md) | 数学の基礎（全16章） | *Math Is All You Need* |
| [第3巻](Volume3/README.md) | ニューラルネットの基礎（全12章） | *Layers Are All You Need* |
| [第4巻](Volume4/README.md) | 言語をベクトルにする／系列（全12章） | *Tokens Are All You Need* |
| [第5巻](Volume5/README.md) | Transformer の構成要素（全13章） | *Heads Are All You Need* |
| [第6巻](Volume6/README.md) | 『Attention Is All You Need』精読（全11章） | **Attention Is All You Need** |
| [第7巻](Volume7/README.md) | PyTorch で Transformer を実装（全14章） | *Code Is All You Need* |
| [第8巻](Volume8/README.md) | 現代の LLM へ（全12章） | *Scale Is All You Need* |

## このシリーズのねらい

最終的に次の式を読み、PyTorch のコードに落とせるようになることを目指します。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

そのために、数式を「意味・shape・実装の対応」の3点セットで読む姿勢を一貫させています。
