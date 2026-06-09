# 用語集（第6巻）

この巻は論文の用語を扱います。各部品の詳しい意味は第5巻の用語集も参照してください。

- **Transformer**: 再帰も畳み込みも使わず、Attention に基づく系列変換モデル。論文が提案した構造。
- **Encoder–Decoder**: 入力系列を別の出力系列に変換する構造。Encoder が入力を理解し、Decoder が出力を生成する。
- **Encoder ブロック**: Self-Attention（mask なし）＋ FFN を Add & Norm で包んだブロック。
- **Decoder ブロック**: Masked Self-Attention ＋ Cross-Attention ＋ FFN を Add & Norm で包んだブロック。
- **Self-Attention**: Q・K・V を同じ系列から作る Attention（第5巻5章）。
- **Cross-Attention（Encoder–Decoder attention）**: Q を Decoder、K・V を Encoder の出力から作る Attention。
- **Masked Self-Attention**: causal mask で未来を見ないようにした Self-Attention（第5巻6章）。Decoder で使う。
- **Scaled Dot-Product Attention**: `softmax(QK^T / sqrt(d_k)) V`。論文の中心式（第5巻4章）。
- **Multi-Head Attention**: 複数のヘッドで異なる部分空間・位置に同時注目する仕組み（第5巻7章）。
- **Add & Norm**: 残差接続＋LayerNorm。サブlayer を包む（第3巻・第5巻11章）。
- **Positional Encoding**: 順序の情報を与える位置エンコーディング。論文は sin/cos（第5巻8〜9章）。
- **Position-wise Feed-Forward**: 位置ごとに独立に適用する2層 MLP（第5巻10章）。
- **経路長（path length）**: 離れた2位置の情報が影響し合うまでの計算ステップ数。Self-Attention では定数。
- **BLEU**: 機械翻訳の品質を測る指標。論文の結果表で使われる。
- **label smoothing**: 正解を完全な1とせず、わずかに他へ確率を分け、自信過剰を防ぐ正則化。
- **ウォームアップ（warmup）**: 学習初期に学習率を徐々に上げるスケジュール（第3巻11章）。
- **N（スタック数）**: Encoder/Decoder で同じブロックを積む段数。論文では N=6。
- **arXiv:1706.03762**: 論文『Attention Is All You Need』(Vaswani et al., 2017) の識別子。
