# 第7巻　PyTorch で Transformer を実装 — *Code Is All You Need*

[README に戻る](README.md) / [用語集を見る](Glossary.md)

## [第1章　この巻の地図](Chapter%201%20Map%20of%20This%20Volume.md)

- [1.1　この巻で作るもの](Chapter%201%20Map%20of%20This%20Volume.md#11-この巻で作るもの)
- [1.2　なぜ「素手 → PyTorch」の二部構成を取るのか](Chapter%201%20Map%20of%20This%20Volume.md#12-なぜ「素手-→-pytorch」の二部構成を取るのか)
- [1.3　素手パートの出口（釘で固定）](Chapter%201%20Map%20of%20This%20Volume.md#13-素手パートの出口釘で固定)
- [1.4　この巻でやらないこと](Chapter%201%20Map%20of%20This%20Volume.md#14-この巻でやらないこと)
- [1.5　本章のまとめ](Chapter%201%20Map%20of%20This%20Volume.md#15-本章のまとめ)

## [第2章　素手で組む（NumPy だけ）](Chapter%202%20By%20Hand%20with%20NumPy.md)

- [2.1　今回作る小さなネットワーク（1個）](Chapter%202%20By%20Hand%20with%20NumPy.md#21-今回作る小さなネットワーク1個)
- [2.2　順伝播を NumPy で書く](Chapter%202%20By%20Hand%20with%20NumPy.md#22-順伝播を-numpy-で書く)
- [2.3　損失を計算する](Chapter%202%20By%20Hand%20with%20NumPy.md#23-損失を計算する)
- [2.4　勾配を手で導いて書く（連鎖律の回収）](Chapter%202%20By%20Hand%20with%20NumPy.md#24-勾配を手で導いて書く連鎖律の回収)
- [2.5　一周を回して学習させ切る](Chapter%202%20By%20Hand%20with%20NumPy.md#25-一周を回して学習させ切る)
- [2.6　本章のまとめ](Chapter%202%20By%20Hand%20with%20NumPy.md#26-本章のまとめ)

## [第3章　同じネットを PyTorch で書き直す](Chapter%203%20Rewriting%20in%20PyTorch.md)

- [3.1　テンソルに置き換える](Chapter%203%20Rewriting%20in%20PyTorch.md#31-テンソルに置き換える)
- [3.2　`loss.backward()` 一行で逆伝播が消える](Chapter%203%20Rewriting%20in%20PyTorch.md#32-lossbackward-一行で逆伝播が消える)
- [3.3　素手版と数値を突き合わせる](Chapter%203%20Rewriting%20in%20PyTorch.md#33-素手版と数値を突き合わせる)
- [3.4　対比から得られる納得](Chapter%203%20Rewriting%20in%20PyTorch.md#34-対比から得られる納得)
- [3.5　本章のまとめ](Chapter%203%20Rewriting%20in%20PyTorch.md#35-本章のまとめ)

## [第4章　テンソルと autograd](Chapter%204%20Tensors%20and%20Autograd.md)

- [4.1　テンソルの基本操作](Chapter%204%20Tensors%20and%20Autograd.md#41-テンソルの基本操作)
- [4.2　requires_grad と計算グラフ](Chapter%204%20Tensors%20and%20Autograd.md#42-requires_grad-と計算グラフ)
- [4.3　勾配の取り出しと zero_grad](Chapter%204%20Tensors%20and%20Autograd.md#43-勾配の取り出しと-zero_grad)
- [4.4　no_grad と detach（第2巻12章の回収）](Chapter%204%20Tensors%20and%20Autograd.md#44-no_grad-と-detach第2巻12章の回収)
- [4.5　本章のまとめ](Chapter%204%20Tensors%20and%20Autograd.md#45-本章のまとめ)

## [第5章　nn.Module で部品を書く](Chapter%205%20Writing%20Parts%20as%20nn%20Module.md)

- [5.1　`nn.Module` の作法](Chapter%205%20Writing%20Parts%20as%20nn%20Module.md#51-nnmodule-の作法)
- [5.2　パラメータの登録と forward](Chapter%205%20Writing%20Parts%20as%20nn%20Module.md#52-パラメータの登録と-forward)
- [5.3　第5巻の部品を組み直す方針](Chapter%205%20Writing%20Parts%20as%20nn%20Module.md#53-第5巻の部品を組み直す方針)
- [5.4　本章のまとめ](Chapter%205%20Writing%20Parts%20as%20nn%20Module.md#54-本章のまとめ)

## [第6章　データを用意する（文字レベル）](Chapter%206%20Preparing%20Character%20Level%20Data.md)

- [6.1　テキストを文字 ID 列にする（第4巻の回収）](Chapter%206%20Preparing%20Character%20Level%20Data.md#61-テキストを文字-id-列にする第4巻の回収)
- [6.2　入力と「次の文字」ターゲットを作る](Chapter%206%20Preparing%20Character%20Level%20Data.md#62-入力と「次の文字」ターゲットを作る)
- [6.3　バッチとブロック長](Chapter%206%20Preparing%20Character%20Level%20Data.md#63-バッチとブロック長)
- [6.4　DataLoader 的な仕組み](Chapter%206%20Preparing%20Character%20Level%20Data.md#64-dataloader-的な仕組み)
- [6.5　本章のまとめ](Chapter%206%20Preparing%20Character%20Level%20Data.md#65-本章のまとめ)

## [第7章　埋め込みと位置エンコーディングを組む](Chapter%207%20Embedding%20and%20Positional%20Encoding.md)

- [7.1　トークン埋め込み層](Chapter%207%20Embedding%20and%20Positional%20Encoding.md#71-トークン埋め込み層)
- [7.2　位置エンコーディングを足す（第5巻8〜9章の回収）](Chapter%207%20Embedding%20and%20Positional%20Encoding.md#72-位置エンコーディングを足す第5巻8〜9章の回収)
- [7.3　入力表現を確かめる（shape）](Chapter%207%20Embedding%20and%20Positional%20Encoding.md#73-入力表現を確かめるshape)
- [7.4　本章のまとめ](Chapter%207%20Embedding%20and%20Positional%20Encoding.md#74-本章のまとめ)

## [第8章　Attention 部品を nn.Module にする](Chapter%208%20Attention%20as%20nn%20Module.md)

- [8.1　第5巻の Attention を持ってくる](Chapter%208%20Attention%20as%20nn%20Module.md#81-第5巻の-attention-を持ってくる)
- [8.2　causal mask 付き Multi-Head Attention](Chapter%208%20Attention%20as%20nn%20Module.md#82-causal-mask-付き-multi-head-attention)
- [8.3　register_buffer とは](Chapter%208%20Attention%20as%20nn%20Module.md#83-register_buffer-とは)
- [8.4　動作と shape を確かめる](Chapter%208%20Attention%20as%20nn%20Module.md#84-動作と-shape-を確かめる)
- [8.5　本章のまとめ](Chapter%208%20Attention%20as%20nn%20Module.md#85-本章のまとめ)

## [第9章　Transformer ブロックを組む](Chapter%209%20Building%20the%20Transformer%20Block.md)

- [9.1　FFN を用意する](Chapter%209%20Building%20the%20Transformer%20Block.md#91-ffn-を用意する)
- [9.2　Pre-LN を採用する理由](Chapter%209%20Building%20the%20Transformer%20Block.md#92-pre-ln-を採用する理由)
- [9.3　ブロックを `nn.Module` にまとめる](Chapter%209%20Building%20the%20Transformer%20Block.md#93-ブロックを-nnmodule-にまとめる)
- [9.4　ブロックを積む](Chapter%209%20Building%20the%20Transformer%20Block.md#94-ブロックを積む)
- [9.5　本章のまとめ](Chapter%209%20Building%20the%20Transformer%20Block.md#95-本章のまとめ)

## [第10章　モデル全体を組む（小さな GPT）](Chapter%2010%20Assembling%20a%20Small%20GPT.md)

- [10.1　embedding → ブロック列 → 出力層](Chapter%2010%20Assembling%20a%20Small%20GPT.md#101-embedding-→-ブロック列-→-出力層)
- [10.2　モデル全体を書く](Chapter%2010%20Assembling%20a%20Small%20GPT.md#102-モデル全体を書く)
- [10.3　出力を語彙上の分布にする（損失の shape）](Chapter%2010%20Assembling%20a%20Small%20GPT.md#103-出力を語彙上の分布にする損失の-shape)
- [10.4　モデルのパラメータ数を確かめる](Chapter%2010%20Assembling%20a%20Small%20GPT.md#104-モデルのパラメータ数を確かめる)
- [10.5　本章のまとめ](Chapter%2010%20Assembling%20a%20Small%20GPT.md#105-本章のまとめ)

## [第11章　学習ループを書く](Chapter%2011%20Writing%20the%20Training%20Loop.md)

- [11.1　学習ループの骨格は変わらない](Chapter%2011%20Writing%20the%20Training%20Loop.md#111-学習ループの骨格は変わらない)
- [11.2　optimizer（AdamW）と学習率](Chapter%2011%20Writing%20the%20Training%20Loop.md#112-optimizeradamwと学習率)
- [11.3　学習ループを回す](Chapter%2011%20Writing%20the%20Training%20Loop.md#113-学習ループを回す)
- [11.4　損失が下がるのを確認する](Chapter%2011%20Writing%20the%20Training%20Loop.md#114-損失が下がるのを確認する)
- [11.5　よくある詰まり](Chapter%2011%20Writing%20the%20Training%20Loop.md#115-よくある詰まり)
- [11.6　本章のまとめ](Chapter%2011%20Writing%20the%20Training%20Loop.md#116-本章のまとめ)

## [第12章　文字を生成させる](Chapter%2012%20Generating%20Text.md)

- [12.1　次の文字を確率分布から選ぶ](Chapter%2012%20Generating%20Text.md#121-次の文字を確率分布から選ぶ)
- [12.2　temperature とサンプリング（第2巻8章の回収）](Chapter%2012%20Generating%20Text.md#122-temperature-とサンプリング第2巻8章の回収)
- [12.3　文章を1文字ずつ伸ばす](Chapter%2012%20Generating%20Text.md#123-文章を1文字ずつ伸ばす)
- [12.4　学習前後で出力を比べる](Chapter%2012%20Generating%20Text.md#124-学習前後で出力を比べる)
- [12.5　本章のまとめ](Chapter%2012%20Generating%20Text.md#125-本章のまとめ)

## [第13章　デバッグと shape 地獄の歩き方](Chapter%2013%20Debugging%20and%20Shape%20Hell.md)

- [13.1　shape を常に印字する習慣](Chapter%2013%20Debugging%20and%20Shape%20Hell.md#131-shape-を常に印字する習慣)
- [13.2　mask が効いているかを確かめる](Chapter%2013%20Debugging%20and%20Shape%20Hell.md#132-mask-が効いているかを確かめる)
- [13.3　勾配が流れているかを確かめる](Chapter%2013%20Debugging%20and%20Shape%20Hell.md#133-勾配が流れているかを確かめる)
- [13.4　小さく作って大きくする](Chapter%2013%20Debugging%20and%20Shape%20Hell.md#134-小さく作って大きくする)
- [13.5　本章のまとめ](Chapter%2013%20Debugging%20and%20Shape%20Hell.md#135-本章のまとめ)

## [第14章　まとめ：自分で Transformer を動かせた](Chapter%2014%20Summary.md)

- [14.1　素手から始めてここまで来た道のり](Chapter%2014%20Summary.md#141-素手から始めてここまで来た道のり)
- [14.2　いま動いているものは論文のどこに対応するか](Chapter%2014%20Summary.md#142-いま動いているものは論文のどこに対応するか)
- [14.3　次の巻（現代の LLM へ）への橋](Chapter%2014%20Summary.md#143-次の巻現代の-llm-へへの橋)
- [14.4　本章のまとめ](Chapter%2014%20Summary.md#144-本章のまとめ)
