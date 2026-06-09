# 第6巻　『Attention Is All You Need』精読 — *Attention Is All You Need*

[README に戻る](README.md) / [用語集を見る](Glossary.md)

## [第1章　この巻の読み方](Chapter%201%20How%20to%20Read%20This%20Volume.md)

- [1.1　「全部もう知っている」状態で論文を開く](Chapter%201%20How%20to%20Read%20This%20Volume.md#11-「全部もう知っている」状態で論文を開く)
- [1.2　第5巻の部品と論文の式の対応表](Chapter%201%20How%20to%20Read%20This%20Volume.md#12-第5巻の部品と論文の式の対応表)
- [1.3　見るべきは「提示順」と「設計理由」](Chapter%201%20How%20to%20Read%20This%20Volume.md#13-見るべきは「提示順」と「設計理由」)
- [1.4　本章のまとめ](Chapter%201%20How%20to%20Read%20This%20Volume.md#14-本章のまとめ)

## [第2章　Abstract と Introduction を読む](Chapter%202%20Reading%20the%20Abstract%20and%20Introduction.md)

- [2.1　Abstract の主張を自分の言葉で](Chapter%202%20Reading%20the%20Abstract%20and%20Introduction.md#21-abstract-の主張を自分の言葉で)
- [2.2　Attention だけで系列を扱うという主張](Chapter%202%20Reading%20the%20Abstract%20and%20Introduction.md#22-attention-だけで系列を扱うという主張)
- [2.3　当時の文脈（何に対する改善か）](Chapter%202%20Reading%20the%20Abstract%20and%20Introduction.md#23-当時の文脈何に対する改善か)
- [2.4　本章のまとめ](Chapter%202%20Reading%20the%20Abstract%20and%20Introduction.md#24-本章のまとめ)

## [第3章　Model Architecture：全体図を自分の言葉で](Chapter%203%20Model%20Architecture%20in%20Your%20Own%20Words.md)

- [3.1　Encoder–Decoder の全体像](Chapter%203%20Model%20Architecture%20in%20Your%20Own%20Words.md#31-encoder–decoder-の全体像)
- [3.2　論文の図1を、第5巻の部品で読み替える](Chapter%203%20Model%20Architecture%20in%20Your%20Own%20Words.md#32-論文の図1を第5巻の部品で読み替える)
- [3.3　入力から出力までの流れ](Chapter%203%20Model%20Architecture%20in%20Your%20Own%20Words.md#33-入力から出力までの流れ)
- [3.4　本章のまとめ](Chapter%203%20Model%20Architecture%20in%20Your%20Own%20Words.md#34-本章のまとめ)

## [第4章　Encoder と Decoder のスタック](Chapter%204%20The%20Encoder%20and%20Decoder%20Stacks.md)

- [4.1　同じブロックを積む構造](Chapter%204%20The%20Encoder%20and%20Decoder%20Stacks.md#41-同じブロックを積む構造)
- [4.2　Encoder ブロックの中身](Chapter%204%20The%20Encoder%20and%20Decoder%20Stacks.md#42-encoder-ブロックの中身)
- [4.3　Decoder ブロックの中身（masked self-attention）](Chapter%204%20The%20Encoder%20and%20Decoder%20Stacks.md#43-decoder-ブロックの中身masked-self-attention)
- [4.4　Encoder–Decoder attention（cross-attention）](Chapter%204%20The%20Encoder%20and%20Decoder%20Stacks.md#44-encoder–decoder-attentioncross-attention)
- [4.5　本章のまとめ](Chapter%204%20The%20Encoder%20and%20Decoder%20Stacks.md#45-本章のまとめ)

## [第5章　Attention の節を読む](Chapter%205%20Reading%20the%20Attention%20Section.md)

- [5.1　Scaled Dot-Product Attention の式](Chapter%205%20Reading%20the%20Attention%20Section.md#51-scaled-dot-product-attention-の式)
- [5.2　`sqrt(d_k)` の説明を著者はどう書いているか](Chapter%205%20Reading%20the%20Attention%20Section.md#52-sqrtd_k-の説明を著者はどう書いているか)
- [5.3　mask の扱い](Chapter%205%20Reading%20the%20Attention%20Section.md#53-mask-の扱い)
- [5.4　本章のまとめ](Chapter%205%20Reading%20the%20Attention%20Section.md#54-本章のまとめ)

## [第6章　Multi-Head Attention の節を読む](Chapter%206%20Reading%20the%20Multi-Head%20Section.md)

- [6.1　なぜ複数ヘッドかの著者の説明](Chapter%206%20Reading%20the%20Multi-Head%20Section.md#61-なぜ複数ヘッドかの著者の説明)
- [6.2　次元の分割と結合（第5巻7章の対応）](Chapter%206%20Reading%20the%20Multi-Head%20Section.md#62-次元の分割と結合第5巻7章の対応)
- [6.3　計算量の話](Chapter%206%20Reading%20the%20Multi-Head%20Section.md#63-計算量の話)
- [6.4　本章のまとめ](Chapter%206%20Reading%20the%20Multi-Head%20Section.md#64-本章のまとめ)

## [第7章　位置エンコーディングの節を読む](Chapter%207%20Reading%20the%20Positional%20Encoding%20Section.md)

- [7.1　なぜ位置エンコーディングが要るのか（論文の論理）](Chapter%207%20Reading%20the%20Positional%20Encoding%20Section.md#71-なぜ位置エンコーディングが要るのか論文の論理)
- [7.2　sin / cos を選んだ理由](Chapter%207%20Reading%20the%20Positional%20Encoding%20Section.md#72-sin-cos-を選んだ理由)
- [7.3　学習埋め込みとの比較を著者はどう述べたか](Chapter%207%20Reading%20the%20Positional%20Encoding%20Section.md#73-学習埋め込みとの比較を著者はどう述べたか)
- [7.4　本章のまとめ](Chapter%207%20Reading%20the%20Positional%20Encoding%20Section.md#74-本章のまとめ)

## [第8章　Why Self-Attention（設計理由）](Chapter%208%20Why%20Self-Attention.md)

- [8.1　3つの観点](Chapter%208%20Why%20Self-Attention.md#81-3つの観点)
- [8.2　経路長という考え方](Chapter%208%20Why%20Self-Attention.md#82-経路長という考え方)
- [8.3　再帰・畳み込みとの比較表を読む](Chapter%208%20Why%20Self-Attention.md#83-再帰・畳み込みとの比較表を読む)
- [8.4　なぜこの設計が効くのか](Chapter%208%20Why%20Self-Attention.md#84-なぜこの設計が効くのか)
- [8.5　本章のまとめ](Chapter%208%20Why%20Self-Attention.md#85-本章のまとめ)

## [第9章　学習設定を読む](Chapter%209%20Reading%20the%20Training%20Setup.md)

- [9.1　この節は第1〜4巻で読める](Chapter%209%20Reading%20the%20Training%20Setup.md#91-この節は第1〜4巻で読める)
- [9.2　最適化（Adam とウォームアップ schedule）](Chapter%209%20Reading%20the%20Training%20Setup.md#92-最適化adam-とウォームアップ-schedule)
- [9.3　正則化（Dropout・label smoothing）](Chapter%209%20Reading%20the%20Training%20Setup.md#93-正則化dropout・label-smoothing)
- [9.4　データとバッチの作り方](Chapter%209%20Reading%20the%20Training%20Setup.md#94-データとバッチの作り方)
- [9.5　第1巻〜第4巻の知識でここが読めることの確認](Chapter%209%20Reading%20the%20Training%20Setup.md#95-第1巻〜第4巻の知識でここが読めることの確認)
- [9.6　本章のまとめ](Chapter%209%20Reading%20the%20Training%20Setup.md#96-本章のまとめ)

## [第10章　結果と結論を読む](Chapter%2010%20Reading%20the%20Results%20and%20Conclusion.md)

- [10.1　翻訳タスクでの結果の読み方](Chapter%2010%20Reading%20the%20Results%20and%20Conclusion.md#101-翻訳タスクでの結果の読み方)
- [10.2　何が示せたと著者は主張しているか](Chapter%2010%20Reading%20the%20Results%20and%20Conclusion.md#102-何が示せたと著者は主張しているか)
- [10.3　限界と今後（論文時点での展望）](Chapter%2010%20Reading%20the%20Results%20and%20Conclusion.md#103-限界と今後論文時点での展望)
- [10.4　本章のまとめ](Chapter%2010%20Reading%20the%20Results%20and%20Conclusion.md#104-本章のまとめ)

## [第11章　まとめ："Attention Is All You Need" の回収](Chapter%2011%20Summary.md)

- [11.1　論文を一周して見えたこと](Chapter%2011%20Summary.md#111-論文を一周して見えたこと)
- [11.2　タイトルの主張は本当に "all you need" だったか](Chapter%2011%20Summary.md#112-タイトルの主張は本当に-all-you-need-だったか)
- [11.3　次の巻（自分で実装して動かす）への橋](Chapter%2011%20Summary.md#113-次の巻自分で実装して動かすへの橋)
- [11.4　本章のまとめ](Chapter%2011%20Summary.md#114-本章のまとめ)
