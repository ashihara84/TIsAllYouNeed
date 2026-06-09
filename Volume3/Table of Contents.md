# 第3巻　ニューラルネットの基礎 — *Layers Are All You Need*

[README に戻る](README.md) / [用語集を見る](Glossary.md)

## [第1章　この巻の地図](Chapter%201%20Map%20of%20This%20Volume.md)

- [1.1　この巻でできるようになること](Chapter%201%20Map%20of%20This%20Volume.md#11-この巻でできるようになること)
- [1.2　第1巻12章の「直感」を実体化する](Chapter%201%20Map%20of%20This%20Volume.md#12-第1巻12章の「直感」を実体化する)
- [1.3　この巻でやらないこと（＝次の巻の仕事）](Chapter%201%20Map%20of%20This%20Volume.md#13-この巻でやらないこと＝次の巻の仕事)
- [1.4　使う道具（PyTorch を最小限から）](Chapter%201%20Map%20of%20This%20Volume.md#14-使う道具pytorch-を最小限から)
- [1.5　本章のまとめ](Chapter%201%20Map%20of%20This%20Volume.md#15-本章のまとめ)

## [第2章　線形モデルから多層へ](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md)

- [2.1　1個のニューロンを式で書く](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#21-1個のニューロンを式で書く)
- [2.2　線形モデルだけでは解けない例](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#22-線形モデルだけでは解けない例)
- [2.3　層を重ねるとは何か](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#23-層を重ねるとは何か)
- [2.4　活性化関数で「折り曲げる」](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#24-活性化関数で「折り曲げる」)
- [2.5　隠れ層という考え方](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#25-隠れ層という考え方)
- [2.6　多層パーセプトロン（MLP）の全体像](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#26-多層パーセプトロンmlpの全体像)
- [2.7　本章のまとめ](Chapter%202%20From%20Linear%20Models%20to%20Multiple%20Layers.md#27-本章のまとめ)

## [第3章　順伝播を行列で書く](Chapter%203%20Forward%20Propagation%20with%20Matrices.md)

- [3.1　入力ベクトルから出力ベクトルへ](Chapter%203%20Forward%20Propagation%20with%20Matrices.md#31-入力ベクトルから出力ベクトルへ)
- [3.2　重み行列とバイアス（第2巻の線形変換の回収）](Chapter%203%20Forward%20Propagation%20with%20Matrices.md#32-重み行列とバイアス第2巻の線形変換の回収)
- [3.3　1層の順伝播を実装する](Chapter%203%20Forward%20Propagation%20with%20Matrices.md#33-1層の順伝播を実装する)
- [3.4　複数層をつなぐ](Chapter%203%20Forward%20Propagation%20with%20Matrices.md#34-複数層をつなぐ)
- [3.5　バッチをまとめて流す（shape を追う）](Chapter%203%20Forward%20Propagation%20with%20Matrices.md#35-バッチをまとめて流すshape-を追う)
- [3.6　本章のまとめ](Chapter%203%20Forward%20Propagation%20with%20Matrices.md#36-本章のまとめ)

## [第4章　活性化関数](Chapter%204%20Activation%20Functions.md)

- [4.1　なぜ非線形性が要るのか](Chapter%204%20Activation%20Functions.md#41-なぜ非線形性が要るのか)
- [4.2　ReLU](Chapter%204%20Activation%20Functions.md#42-relu)
- [4.3　シグモイドと tanh](Chapter%204%20Activation%20Functions.md#43-シグモイドと-tanh)
- [4.4　活性化関数の選び方の直感](Chapter%204%20Activation%20Functions.md#44-活性化関数の選び方の直感)
- [4.5　活性化関数を順伝播に挟む（実装）](Chapter%204%20Activation%20Functions.md#45-活性化関数を順伝播に挟む実装)
- [4.6　本章のまとめ](Chapter%204%20Activation%20Functions.md#46-本章のまとめ)

## [第5章　出力層と損失](Chapter%205%20Output%20Layers%20and%20Loss.md)

- [5.1　分類の出力は softmax（第2巻の回収・実装で）](Chapter%205%20Output%20Layers%20and%20Loss.md#51-分類の出力は-softmax第2巻の回収・実装で)
- [5.2　cross entropy を損失にする](Chapter%205%20Output%20Layers%20and%20Loss.md#52-cross-entropy-を損失にする)
- [5.3　出力層と損失をつなぐ](Chapter%205%20Output%20Layers%20and%20Loss.md#53-出力層と損失をつなぐ)
- [5.4　1ステップ分の損失を計算する（実装）](Chapter%205%20Output%20Layers%20and%20Loss.md#54-1ステップ分の損失を計算する実装)
- [5.5　本章のまとめ](Chapter%205%20Output%20Layers%20and%20Loss.md#55-本章のまとめ)

## [第6章　逆伝播と連鎖律（実装で）](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md)

- [6.1　学習とは損失を下げる方向に重みを動かすこと](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md#61-学習とは損失を下げる方向に重みを動かすこと)
- [6.2　連鎖律で勾配が後ろへ流れる（第2巻の回収）](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md#62-連鎖律で勾配が後ろへ流れる第2巻の回収)
- [6.3　PyTorch の autograd に任せる](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md#63-pytorch-の-autograd-に任せる)
- [6.4　`loss.backward()` で何が起きているか](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md#64-lossbackward-で何が起きているか)
- [6.5　勾配をのぞいて確かめる](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md#65-勾配をのぞいて確かめる)
- [6.6　本章のまとめ](Chapter%206%20Backpropagation%20and%20the%20Chain%20Rule.md#66-本章のまとめ)

## [第7章　勾配降下で学習させる](Chapter%207%20Training%20with%20Gradient%20Descent.md)

- [7.1　optimizer を用意する（SGD）](Chapter%207%20Training%20with%20Gradient%20Descent.md#71-optimizer-を用意するsgd)
- [7.2　学習ループの骨格（forward → loss → backward → step）](Chapter%207%20Training%20with%20Gradient%20Descent.md#72-学習ループの骨格forward-→-loss-→-backward-→-step)
- [7.3　学習率とエポックの実際](Chapter%207%20Training%20with%20Gradient%20Descent.md#73-学習率とエポックの実際)
- [7.4　Adam / AdamW に切り替える](Chapter%207%20Training%20with%20Gradient%20Descent.md#74-adam-adamw-に切り替える)
- [7.5　学習が進んでいるかを損失で見る](Chapter%207%20Training%20with%20Gradient%20Descent.md#75-学習が進んでいるかを損失で見る)
- [7.6　本章のまとめ](Chapter%207%20Training%20with%20Gradient%20Descent.md#76-本章のまとめ)

## [第8章　ミニ MLP を最後まで学習させ切る](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md)

- [8.1　小さなデータセットを用意する](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md#81-小さなデータセットを用意する)
- [8.2　モデルを `nn.Module` で書く](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md#82-モデルを-nnmodule-で書く)
- [8.3　学習ループを回す](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md#83-学習ループを回す)
- [8.4　学習結果を確認する](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md#84-学習結果を確認する)
- [8.5　うまくいかないときの最初のチェック](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md#85-うまくいかないときの最初のチェック)
- [8.6　本章のまとめ](Chapter%208%20Training%20a%20Mini%20MLP%20End%20to%20End.md#86-本章のまとめ)

## [第9章　残差接続](Chapter%209%20Residual%20Connections.md)

- [9.1　層を深くすると起きる問題](Chapter%209%20Residual%20Connections.md#91-層を深くすると起きる問題)
- [9.2　残差接続とは何か（`x + f(x)`）](Chapter%209%20Residual%20Connections.md#92-残差接続とは何かx-+-fx)
- [9.3　なぜ勾配が流れやすくなるのか（数で確かめる）](Chapter%209%20Residual%20Connections.md#93-なぜ勾配が流れやすくなるのか数で確かめる)
- [9.4　残差接続を実装する](Chapter%209%20Residual%20Connections.md#94-残差接続を実装する)
- [9.5　論文の `x + Sublayer(x)` への布石](Chapter%209%20Residual%20Connections.md#95-論文の-x-+-sublayerx-への布石)
- [9.6　本章のまとめ](Chapter%209%20Residual%20Connections.md#96-本章のまとめ)

## [第10章　Layer Normalization](Chapter%2010%20Layer%20Normalization.md)

- [10.1　層をまたぐと値のスケールが暴れる](Chapter%2010%20Layer%20Normalization.md#101-層をまたぐと値のスケールが暴れる)
- [10.2　正規化の復習（第2巻の回収）](Chapter%2010%20Layer%20Normalization.md#102-正規化の復習第2巻の回収)
- [10.3　LayerNorm とは何か（数で確かめる）](Chapter%2010%20Layer%20Normalization.md#103-layernorm-とは何か数で確かめる)
- [10.4　gamma と beta の役割](Chapter%2010%20Layer%20Normalization.md#104-gamma-と-beta-の役割)
- [10.5　LayerNorm を実装する](Chapter%2010%20Layer%20Normalization.md#105-layernorm-を実装する)
- [10.6　残差接続と LayerNorm を組み合わせる](Chapter%2010%20Layer%20Normalization.md#106-残差接続と-layernorm-を組み合わせる)
- [10.7　論文の `LayerNorm(x + Sublayer(x))` を読み解く準備](Chapter%2010%20Layer%20Normalization.md#107-論文の-layernormx-+-sublayerx-を読み解く準備)
- [10.8　本章のまとめ](Chapter%2010%20Layer%20Normalization.md#108-本章のまとめ)

## [第11章　学習を安定させる小技](Chapter%2011%20Techniques%20for%20Stable%20Training.md)

- [11.1　重みの初期化](Chapter%2011%20Techniques%20for%20Stable%20Training.md#111-重みの初期化)
- [11.2　学習率のウォームアップと減衰の直感](Chapter%2011%20Techniques%20for%20Stable%20Training.md#112-学習率のウォームアップと減衰の直感)
- [11.3　過学習への基本対処（Dropout を実装で）](Chapter%2011%20Techniques%20for%20Stable%20Training.md#113-過学習への基本対処dropout-を実装で)
- [11.4　勾配が爆発・消失するとき](Chapter%2011%20Techniques%20for%20Stable%20Training.md#114-勾配が爆発・消失するとき)
- [11.5　本章のまとめ](Chapter%2011%20Techniques%20for%20Stable%20Training.md#115-本章のまとめ)

## [第12章　まとめ：論文の一行を謎にしないために](Chapter%2012%20Summary.md)

- [12.1　この巻で書けるようになったもの（部品表）](Chapter%2012%20Summary.md#121-この巻で書けるようになったもの部品表)
- [12.2　残差接続と LayerNorm はこの先どこで効くか](Chapter%2012%20Summary.md#122-残差接続と-layernorm-はこの先どこで効くか)
- [12.3　次の巻（言語をベクトルにする）への橋](Chapter%2012%20Summary.md#123-次の巻言語をベクトルにするへの橋)
- [12.4　本章のまとめ](Chapter%2012%20Summary.md#124-本章のまとめ)
