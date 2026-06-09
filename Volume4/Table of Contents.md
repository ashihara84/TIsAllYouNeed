# 第4巻　言語をベクトルにする／系列を扱う — *Tokens Are All You Need*

[README に戻る](README.md) / [用語集を見る](Glossary.md)

## [第1章　この巻の地図](Chapter%201%20Map%20of%20This%20Volume.md)

- [1.1　言語を扱うと何が新しいのか](Chapter%201%20Map%20of%20This%20Volume.md#11-言語を扱うと何が新しいのか)
- [1.2　この巻でできるようになること](Chapter%201%20Map%20of%20This%20Volume.md#12-この巻でできるようになること)
- [1.3　この巻でやらないこと（＝次の巻の仕事）](Chapter%201%20Map%20of%20This%20Volume.md#13-この巻でやらないこと＝次の巻の仕事)
- [1.4　本章のまとめ](Chapter%201%20Map%20of%20This%20Volume.md#14-本章のまとめ)

## [第2章　言語をどう数にするか](Chapter%202%20Turning%20Language%20into%20Numbers.md)

- [2.1　文字・単語・記号をコンピュータに渡す](Chapter%202%20Turning%20Language%20into%20Numbers.md#21-文字・単語・記号をコンピュータに渡す)
- [2.2　離散的な記号と連続的なベクトル](Chapter%202%20Turning%20Language%20into%20Numbers.md#22-離散的な記号と連続的なベクトル)
- [2.3　「意味の近さ」を数で表したい](Chapter%202%20Turning%20Language%20into%20Numbers.md#23-「意味の近さ」を数で表したい)
- [2.4　本章のまとめ](Chapter%202%20Turning%20Language%20into%20Numbers.md#24-本章のまとめ)

## [第3章　トークン化（tokenization）](Chapter%203%20Tokenization.md)

- [3.1　トークンとは何か](Chapter%203%20Tokenization.md#31-トークンとは何か)
- [3.2　文字単位・単語単位・サブワード単位](Chapter%203%20Tokenization.md#32-文字単位・単語単位・サブワード単位)
- [3.3　語彙（vocabulary）と未知語](Chapter%203%20Tokenization.md#33-語彙vocabularyと未知語)
- [3.4　サブワード分割の直感（BPE の考え方）](Chapter%203%20Tokenization.md#34-サブワード分割の直感bpe-の考え方)
- [3.5　トークン化を小さく試す](Chapter%203%20Tokenization.md#35-トークン化を小さく試す)
- [3.6　本章のまとめ](Chapter%203%20Tokenization.md#36-本章のまとめ)

## [第4章　語彙と ID](Chapter%204%20Vocabulary%20and%20IDs.md)

- [4.1　トークンに ID を振る](Chapter%204%20Vocabulary%20and%20IDs.md#41-トークンに-id-を振る)
- [4.2　ID 列としての文章](Chapter%204%20Vocabulary%20and%20IDs.md#42-id-列としての文章)
- [4.3　特殊トークン（開始・終了・パディング）](Chapter%204%20Vocabulary%20and%20IDs.md#43-特殊トークン開始・終了・パディング)
- [4.4　ID 列を作って確かめる](Chapter%204%20Vocabulary%20and%20IDs.md#44-id-列を作って確かめる)
- [4.5　本章のまとめ](Chapter%204%20Vocabulary%20and%20IDs.md#45-本章のまとめ)

## [第5章　embedding：トークンをベクトルにする](Chapter%205%20Embeddings.md)

- [5.1　one-hot の限界（第1巻・第2巻の回収）](Chapter%205%20Embeddings.md#51-one-hot-の限界第1巻・第2巻の回収)
- [5.2　embedding 行列とは何か](Chapter%205%20Embeddings.md#52-embedding-行列とは何か)
- [5.3　ID から表現ベクトルを引く](Chapter%205%20Embeddings.md#53-id-から表現ベクトルを引く)
- [5.4　学習で意味が宿るとはどういうことか](Chapter%205%20Embeddings.md#54-学習で意味が宿るとはどういうことか)
- [5.5　`nn.Embedding` を使って確かめる](Chapter%205%20Embeddings.md#55-nnembedding-を使って確かめる)
- [5.6　本章のまとめ](Chapter%205%20Embeddings.md#56-本章のまとめ)

## [第6章　系列データを扱う](Chapter%206%20Working%20with%20Sequences.md)

- [6.1　系列とは何か（順序が意味を持つ）](Chapter%206%20Working%20with%20Sequences.md#61-系列とは何か順序が意味を持つ)
- [6.2　可変長をどう扱うか](Chapter%206%20Working%20with%20Sequences.md#62-可変長をどう扱うか)
- [6.3　文脈（context）という考え方](Chapter%206%20Working%20with%20Sequences.md#63-文脈contextという考え方)
- [6.4　系列を行列として見る（shape を追う）](Chapter%206%20Working%20with%20Sequences.md#64-系列を行列として見るshape-を追う)
- [6.5　本章のまとめ](Chapter%206%20Working%20with%20Sequences.md#65-本章のまとめ)

## [第7章　言語モデルとは](Chapter%207%20What%20Is%20a%20Language%20Model.md)

- [7.1　次のトークンを予測するというタスク](Chapter%207%20What%20Is%20a%20Language%20Model.md#71-次のトークンを予測するというタスク)
- [7.2　条件付き確率としての言語モデル（第1巻11章の回収）](Chapter%207%20What%20Is%20a%20Language%20Model.md#72-条件付き確率としての言語モデル第1巻11章の回収)
- [7.3　言語モデルの損失（cross entropy・第2巻の回収）](Chapter%207%20What%20Is%20a%20Language%20Model.md#73-言語モデルの損失cross-entropy・第2巻の回収)
- [7.4　生成：確率分布からトークンを選ぶ](Chapter%207%20What%20Is%20a%20Language%20Model.md#74-生成：確率分布からトークンを選ぶ)
- [7.5　本章のまとめ](Chapter%207%20What%20Is%20a%20Language%20Model.md#75-本章のまとめ)

## [第8章　n-gram からニューラル言語モデルへ](Chapter%208%20From%20N-grams%20to%20Neural%20Language%20Models.md)

- [8.1　n-gram の考え方](Chapter%208%20From%20N-grams%20to%20Neural%20Language%20Models.md#81-n-gram-の考え方)
- [8.2　n-gram の限界](Chapter%208%20From%20N-grams%20to%20Neural%20Language%20Models.md#82-n-gram-の限界)
- [8.3　ニューラル言語モデルという発想](Chapter%208%20From%20N-grams%20to%20Neural%20Language%20Models.md#83-ニューラル言語モデルという発想)
- [8.4　固定窓モデルの限界](Chapter%208%20From%20N-grams%20to%20Neural%20Language%20Models.md#84-固定窓モデルの限界)
- [8.5　本章のまとめ](Chapter%208%20From%20N-grams%20to%20Neural%20Language%20Models.md#85-本章のまとめ)

## [第9章　RNN の考え方（実装しない）](Chapter%209%20The%20Idea%20of%20RNNs.md)

- [9.1　状態を持ち越すという発想](Chapter%209%20The%20Idea%20of%20RNNs.md#91-状態を持ち越すという発想)
- [9.2　RNN の基本構造（図と直感）](Chapter%209%20The%20Idea%20of%20RNNs.md#92-rnn-の基本構造図と直感)
- [9.3　時間方向に展開する](Chapter%209%20The%20Idea%20of%20RNNs.md#93-時間方向に展開する)
- [9.4　LSTM / GRU が足したもの（直感）](Chapter%209%20The%20Idea%20of%20RNNs.md#94-lstm-gru-が足したもの直感)
- [9.5　本章のまとめ](Chapter%209%20The%20Idea%20of%20RNNs.md#95-本章のまとめ)

## [第10章　RNN は何が辛いのか](Chapter%2010%20Why%20RNNs%20Are%20Hard.md)

- [10.1　長距離依存をうまく運べない](Chapter%2010%20Why%20RNNs%20Are%20Hard.md#101-長距離依存をうまく運べない)
- [10.2　勾配の消失・爆発](Chapter%2010%20Why%20RNNs%20Are%20Hard.md#102-勾配の消失・爆発)
- [10.3　逐次処理で並列化できない](Chapter%2010%20Why%20RNNs%20Are%20Hard.md#103-逐次処理で並列化できない)
- [10.4　「遠くの単語を直接見たい」という欲求](Chapter%2010%20Why%20RNNs%20Are%20Hard.md#104-「遠くの単語を直接見たい」という欲求)
- [10.5　本章のまとめ](Chapter%2010%20Why%20RNNs%20Are%20Hard.md#105-本章のまとめ)

## [第11章　Attention は何を足したか（動機づけ）](Chapter%2011%20What%20Attention%20Added.md)

- [11.1　すべてのトークンを直接見比べる](Chapter%2011%20What%20Attention%20Added.md#111-すべてのトークンを直接見比べる)
- [11.2　「どこに注目するか」を重みで表す](Chapter%2011%20What%20Attention%20Added.md#112-「どこに注目するか」を重みで表す)
- [11.3　内積で相性を測る（第2巻4章の回収）](Chapter%2011%20What%20Attention%20Added.md#113-内積で相性を測る第2巻4章の回収)
- [11.4　並列に計算できるという利点](Chapter%2011%20What%20Attention%20Added.md#114-並列に計算できるという利点)
- [11.5　ここまでで Attention の必要性が腑に落ちる](Chapter%2011%20What%20Attention%20Added.md#115-ここまでで-attention-の必要性が腑に落ちる)
- [11.6　本章のまとめ](Chapter%2011%20What%20Attention%20Added.md#116-本章のまとめ)

## [第12章　まとめ：5巻への橋](Chapter%2012%20Summary.md)

- [12.1　この巻でたどった道のり](Chapter%2012%20Summary.md#121-この巻でたどった道のり)
- [12.2　いま語れるようになったこと](Chapter%2012%20Summary.md#122-いま語れるようになったこと)
- [12.3　なぜ次に Attention の部品を作るのか](Chapter%2012%20Summary.md#123-なぜ次に-attention-の部品を作るのか)
- [12.4　次の巻への橋](Chapter%2012%20Summary.md#124-次の巻への橋)
- [12.5　本章のまとめ](Chapter%2012%20Summary.md#125-本章のまとめ)
