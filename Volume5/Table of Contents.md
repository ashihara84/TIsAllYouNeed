# 第5巻　Transformer の構成要素 — *Heads Are All You Need*

[README に戻る](README.md) / [用語集を見る](Glossary.md)

## [第1章　この巻の地図と教え方の順番](Chapter%201%20Map%20and%20Teaching%20Order.md)

- [1.1　この巻で作る部品の一覧（部品表）](Chapter%201%20Map%20and%20Teaching%20Order.md#11-この巻で作る部品の一覧部品表)
- [1.2　なぜ論文の順で教えないのか](Chapter%201%20Map%20and%20Teaching%20Order.md#12-なぜ論文の順で教えないのか)
- [1.3　最適な順：核から少しずつ足す](Chapter%201%20Map%20and%20Teaching%20Order.md#13-最適な順：核から少しずつ足す)
- [1.4　この巻でやらないこと（＝次の巻の仕事）](Chapter%201%20Map%20and%20Teaching%20Order.md#14-この巻でやらないこと＝次の巻の仕事)
- [1.5　本章のまとめ](Chapter%201%20Map%20and%20Teaching%20Order.md#15-本章のまとめ)

## [第2章　Attention の核：重み付き和](Chapter%202%20The%20Core%20of%20Attention%20Weighted%20Sum.md)

- [2.1　「注目の重み」で値を混ぜる](Chapter%202%20The%20Core%20of%20Attention%20Weighted%20Sum.md#21-「注目の重み」で値を混ぜる)
- [2.2　重みが合計1になる理由（softmax の回収）](Chapter%202%20The%20Core%20of%20Attention%20Weighted%20Sum.md#22-重みが合計1になる理由softmax-の回収)
- [2.3　最小の重み付き和を実装する](Chapter%202%20The%20Core%20of%20Attention%20Weighted%20Sum.md#23-最小の重み付き和を実装する)
- [2.4　本章のまとめ](Chapter%202%20The%20Core%20of%20Attention%20Weighted%20Sum.md#24-本章のまとめ)

## [第3章　Q / K / V という役割分担](Chapter%203%20The%20Roles%20of%20Q%20K%20V.md)

- [3.1　Query・Key・Value のたとえ](Chapter%203%20The%20Roles%20of%20Q%20K%20V.md#31-query・key・value-のたとえ)
- [3.2　なぜ役割を3つに分けるのか](Chapter%203%20The%20Roles%20of%20Q%20K%20V.md#32-なぜ役割を3つに分けるのか)
- [3.3　embedding から Q / K / V を作る（第2巻6章の回収）](Chapter%203%20The%20Roles%20of%20Q%20K%20V.md#33-embedding-から-q-k-v-を作る第2巻6章の回収)
- [3.4　Q / K / V を実装して shape を確かめる](Chapter%203%20The%20Roles%20of%20Q%20K%20V.md#34-q-k-v-を実装して-shape-を確かめる)
- [3.5　本章のまとめ](Chapter%203%20The%20Roles%20of%20Q%20K%20V.md#35-本章のまとめ)

## [第4章　スケール付き内積 Attention](Chapter%204%20Scaled%20Dot-Product%20Attention.md)

- [4.1　QK^T で相性を測る](Chapter%204%20Scaled%20Dot-Product%20Attention.md#41-qkt-で相性を測る)
- [4.2　`sqrt(d_k)` で割る理由](Chapter%204%20Scaled%20Dot-Product%20Attention.md#42-sqrtd_k-で割る理由)
- [4.3　softmax で重みにし、V を重み付き和にする](Chapter%204%20Scaled%20Dot-Product%20Attention.md#43-softmax-で重みにしv-を重み付き和にする)
- [4.4　中心式を関数として実装する](Chapter%204%20Scaled%20Dot-Product%20Attention.md#44-中心式を関数として実装する)
- [4.5　本章のまとめ](Chapter%204%20Scaled%20Dot-Product%20Attention.md#45-本章のまとめ)

## [第5章　Self-Attention](Chapter%205%20Self-Attention.md)

- [5.1　自分自身を Q・K・V にするとは](Chapter%205%20Self-Attention.md#51-自分自身を-q・k・v-にするとは)
- [5.2　各トークンが系列全体を見る](Chapter%205%20Self-Attention.md#52-各トークンが系列全体を見る)
- [5.3　Self-Attention モジュールを書く](Chapter%205%20Self-Attention.md#53-self-attention-モジュールを書く)
- [5.4　出力の shape と意味を確かめる](Chapter%205%20Self-Attention.md#54-出力の-shape-と意味を確かめる)
- [5.5　本章のまとめ](Chapter%205%20Self-Attention.md#55-本章のまとめ)

## [第6章　Causal mask（未来を見ない）](Chapter%206%20Causal%20Masking.md)

- [6.1　言語モデルでは未来のトークンを見てはいけない](Chapter%206%20Causal%20Masking.md#61-言語モデルでは未来のトークンを見てはいけない)
- [6.2　mask で重みを潰す仕組み](Chapter%206%20Causal%20Masking.md#62-mask-で重みを潰す仕組み)
- [6.3　causal mask を実装する](Chapter%206%20Causal%20Masking.md#63-causal-mask-を実装する)
- [6.4　mask 付き Self-Attention を確かめる](Chapter%206%20Causal%20Masking.md#64-mask-付き-self-attention-を確かめる)
- [6.5　本章のまとめ](Chapter%206%20Causal%20Masking.md#65-本章のまとめ)

## [第7章　Multi-Head Attention](Chapter%207%20Multi-Head%20Attention.md)

- [7.1　なぜ複数のヘッドが要るのか](Chapter%207%20Multi-Head%20Attention.md#71-なぜ複数のヘッドが要るのか)
- [7.2　表現を分割して並列に注目する](Chapter%207%20Multi-Head%20Attention.md#72-表現を分割して並列に注目する)
- [7.3　ヘッドごとの計算と結合](Chapter%207%20Multi-Head%20Attention.md#73-ヘッドごとの計算と結合)
- [7.4　Multi-Head Attention を実装する](Chapter%207%20Multi-Head%20Attention.md#74-multi-head-attention-を実装する)
- [7.5　shape を最後まで追う](Chapter%207%20Multi-Head%20Attention.md#75-shape-を最後まで追う)
- [7.6　本章のまとめ](Chapter%207%20Multi-Head%20Attention.md#76-本章のまとめ)

## [第8章　位置エンコーディング](Chapter%208%20Positional%20Encoding.md)

- [8.1　Attention は順序を知らない](Chapter%208%20Positional%20Encoding.md#81-attention-は順序を知らない)
- [8.2　位置の情報をどう足すか](Chapter%208%20Positional%20Encoding.md#82-位置の情報をどう足すか)
- [8.3　sin / cos による位置エンコーディング](Chapter%208%20Positional%20Encoding.md#83-sin-cos-による位置エンコーディング)
- [8.4　学習する位置埋め込みとの違い（直感）](Chapter%208%20Positional%20Encoding.md#84-学習する位置埋め込みとの違い直感)
- [8.5　本章のまとめ](Chapter%208%20Positional%20Encoding.md#85-本章のまとめ)

## [第9章　位置エンコーディングを書く](Chapter%209%20Implementing%20Positional%20Encoding.md)

- [9.1　sin / cos の位置エンコーディングを実装する](Chapter%209%20Implementing%20Positional%20Encoding.md#91-sin-cos-の位置エンコーディングを実装する)
- [9.2　embedding に足して確かめる](Chapter%209%20Implementing%20Positional%20Encoding.md#92-embedding-に足して確かめる)
- [9.3　学習する位置埋め込みも書いてみる](Chapter%209%20Implementing%20Positional%20Encoding.md#93-学習する位置埋め込みも書いてみる)
- [9.4　本章のまとめ](Chapter%209%20Implementing%20Positional%20Encoding.md#94-本章のまとめ)

## [第10章　Feed-Forward Network（FFN）](Chapter%2010%20Feed-Forward%20Network.md)

- [10.1　Attention の後になぜ FFN が要るのか](Chapter%2010%20Feed-Forward%20Network.md#101-attention-の後になぜ-ffn-が要るのか)
- [10.2　位置ごとに独立な2層 MLP（第3巻の回収）](Chapter%2010%20Feed-Forward%20Network.md#102-位置ごとに独立な2層-mlp第3巻の回収)
- [10.3　次元を広げて戻すという形](Chapter%2010%20Feed-Forward%20Network.md#103-次元を広げて戻すという形)
- [10.4　FFN を実装する](Chapter%2010%20Feed-Forward%20Network.md#104-ffn-を実装する)
- [10.5　本章のまとめ](Chapter%2010%20Feed-Forward%20Network.md#105-本章のまとめ)

## [第11章　残差接続と LayerNorm で部品を包む](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md)

- [11.1　第3巻で仕込んだものの回収](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md#111-第3巻で仕込んだものの回収)
- [11.2　`LayerNorm(x + Sublayer(x))` の形](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md#112-layernormx-+-sublayerx-の形)
- [11.3　Attention と FFN をそれぞれ包む](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md#113-attention-と-ffn-をそれぞれ包む)
- [11.4　Pre-LN と Post-LN の違い（直感）](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md#114-pre-ln-と-post-ln-の違い直感)
- [11.5　本章のまとめ](Chapter%2011%20Wrapping%20with%20Residual%20and%20LayerNorm.md#115-本章のまとめ)

## [第12章　部品を並べてみる（まだ統合しない）](Chapter%2012%20Laying%20Out%20the%20Parts.md)

- [12.1　1つの Transformer ブロックの構成](Chapter%2012%20Laying%20Out%20the%20Parts.md#121-1つの-transformer-ブロックの構成)
- [12.2　ブロックを小さく書いて動かす](Chapter%2012%20Laying%20Out%20the%20Parts.md#122-ブロックを小さく書いて動かす)
- [12.3　なぜ、ここで言語モデルとして通さないのか](Chapter%2012%20Laying%20Out%20the%20Parts.md#123-なぜここで言語モデルとして通さないのか)
- [12.4　本章のまとめ](Chapter%2012%20Laying%20Out%20the%20Parts.md#124-本章のまとめ)

## [第13章　まとめ：部品表と7巻への橋](Chapter%2013%20Summary.md)

- [13.1　作った部品の一覧と shape](Chapter%2013%20Summary.md#131-作った部品の一覧と-shape)
- [13.2　これらが論文のどの式に対応するか（第6巻への橋）](Chapter%2013%20Summary.md#132-これらが論文のどの式に対応するか第6巻への橋)
- [13.3　次の巻で統合して動かす（第7巻への橋）](Chapter%2013%20Summary.md#133-次の巻で統合して動かす第7巻への橋)
- [13.4　本章のまとめ](Chapter%2013%20Summary.md#134-本章のまとめ)
