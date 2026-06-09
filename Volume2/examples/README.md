# Examples

このディレクトリには、本文で扱う数学を小さく実行して確認するためのコードを置いています。

```bash
python3 examples/01_softmax.py
python3 examples/02_cross_entropy.py
python3 examples/03_attention.py
```

各ファイルは、数式が PyTorch のテンソル計算に対応していることを一度確かめるためのものです。部品化・統合は次の巻（5・7巻）の仕事です。

- `01_softmax.py`: softmaxがスコアを重みに変えることを確認する
- `02_cross_entropy.py`: logits、targets、cross entropy lossの関係を確認する
- `03_attention.py`: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` を確認する
