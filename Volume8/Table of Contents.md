# 第8巻　Transformer から現代の LLM へ — *Scale Is All You Need*

[README に戻る](README.md) / [用語集を見る](Glossary.md)

## [第1章　この巻の地図](Chapter%201%20Map%20of%20This%20Volume.md)

- [1.1　元論文の Transformer を出発点に置く](Chapter%201%20Map%20of%20This%20Volume.md#11-元論文の-transformer-を出発点に置く)
- [1.2　現代の LLM を「何を足し、何を削ったか」で読む](Chapter%201%20Map%20of%20This%20Volume.md#12-現代の-llm-を「何を足し何を削ったか」で読む)
- [1.3　この巻でやらないこと](Chapter%201%20Map%20of%20This%20Volume.md#13-この巻でやらないこと)
- [1.4　本章のまとめ](Chapter%201%20Map%20of%20This%20Volume.md#14-本章のまとめ)

## [第2章　元論文から何が変わったか](Chapter%202%20What%20Changed%20from%20the%20Original.md)

- [2.1　全体像：3つの変化](Chapter%202%20What%20Changed%20from%20the%20Original.md#21-全体像：3つの変化)
- [2.2　Encoder–Decoder から Decoder-only へ](Chapter%202%20What%20Changed%20from%20the%20Original.md#22-encoder–decoder-から-decoder-only-へ)
- [2.3　規模の桁が変わった](Chapter%202%20What%20Changed%20from%20the%20Original.md#23-規模の桁が変わった)
- [2.4　学習の目的は同じ（次トークン予測）](Chapter%202%20What%20Changed%20from%20the%20Original.md#24-学習の目的は同じ次トークン予測)
- [2.5　本章のまとめ](Chapter%202%20What%20Changed%20from%20the%20Original.md#25-本章のまとめ)

## [第3章　Decoder-only という選択（GPT 系）](Chapter%203%20The%20Decoder-Only%20Choice.md)

- [3.1　なぜ Decoder だけで足りるのか](Chapter%203%20The%20Decoder-Only%20Choice.md#31-なぜ-decoder-だけで足りるのか)
- [3.2　第7巻で作った小さな GPT がまさにこれである](Chapter%203%20The%20Decoder-Only%20Choice.md#32-第7巻で作った小さな-gpt-がまさにこれである)
- [3.3　Encoder が要るタスク・要らないタスク](Chapter%203%20The%20Decoder-Only%20Choice.md#33-encoder-が要るタスク・要らないタスク)
- [3.4　本章のまとめ](Chapter%203%20The%20Decoder-Only%20Choice.md#34-本章のまとめ)

## [第4章　事前学習とスケール](Chapter%204%20Pretraining%20and%20Scale.md)

- [4.1　事前学習（pretraining）とは](Chapter%204%20Pretraining%20and%20Scale.md#41-事前学習pretrainingとは)
- [4.2　スケーリング則の直感](Chapter%204%20Pretraining%20and%20Scale.md#42-スケーリング則の直感)
- [4.3　何が「創発」して見えるのか（慎重に）](Chapter%204%20Pretraining%20and%20Scale.md#43-何が「創発」して見えるのか慎重に)
- [4.4　本章のまとめ](Chapter%204%20Pretraining%20and%20Scale.md#44-本章のまとめ)

## [第5章　トークナイザの現実](Chapter%205%20Tokenizers%20in%20Practice.md)

- [5.1　現代のトークナイザはサブワード（第4巻3章の回収）](Chapter%205%20Tokenizers%20in%20Practice.md#51-現代のトークナイザはサブワード第4巻3章の回収)
- [5.2　語彙サイズと系列長のトレードオフ](Chapter%205%20Tokenizers%20in%20Practice.md#52-語彙サイズと系列長のトレードオフ)
- [5.3　トークン化が性能・コストに効く理由](Chapter%205%20Tokenizers%20in%20Practice.md#53-トークン化が性能・コストに効く理由)
- [5.4　本章のまとめ](Chapter%205%20Tokenizers%20in%20Practice.md#54-本章のまとめ)

## [第6章　fine-tuning](Chapter%206%20Fine-Tuning.md)

- [6.1　事前学習済みモデルを下流タスクに合わせる](Chapter%206%20Fine-Tuning.md#61-事前学習済みモデルを下流タスクに合わせる)
- [6.2　第1巻の「学習＝損失を下げる」がそのまま効く](Chapter%206%20Fine-Tuning.md#62-第1巻の「学習＝損失を下げる」がそのまま効く)
- [6.3　全パラメータ更新と軽量手法（LoRA などの直感）](Chapter%206%20Fine-Tuning.md#63-全パラメータ更新と軽量手法lora-などの直感)
- [6.4　本章のまとめ](Chapter%206%20Fine-Tuning.md#64-本章のまとめ)

## [第7章　指示チューニング（instruction tuning）](Chapter%207%20Instruction%20Tuning.md)

- [7.1　「次トークン予測」から「指示に従う」へ](Chapter%207%20Instruction%20Tuning.md#71-「次トークン予測」から「指示に従う」へ)
- [7.2　指示データで何を教えているのか](Chapter%207%20Instruction%20Tuning.md#72-指示データで何を教えているのか)
- [7.3　チャット形式の入出力](Chapter%207%20Instruction%20Tuning.md#73-チャット形式の入出力)
- [7.4　本章のまとめ](Chapter%207%20Instruction%20Tuning.md#74-本章のまとめ)

## [第8章　RLHF：人間のフィードバックで整える](Chapter%208%20RLHF.md)

- [8.1　なぜ教師あり学習だけでは足りないのか](Chapter%208%20RLHF.md#81-なぜ教師あり学習だけでは足りないのか)
- [8.2　報酬モデルという考え方](Chapter%208%20RLHF.md#82-報酬モデルという考え方)
- [8.3　人間の好みで方策を更新する（直感）](Chapter%208%20RLHF.md#83-人間の好みで方策を更新する直感)
- [8.4　近年の代替手法の位置づけ（概観）](Chapter%208%20RLHF.md#84-近年の代替手法の位置づけ概観)
- [8.5　本章のまとめ](Chapter%208%20RLHF.md#85-本章のまとめ)

## [第9章　推論時の工夫](Chapter%209%20Inference-Time%20Techniques.md)

- [9.1　サンプリングと temperature（再訪）](Chapter%209%20Inference-Time%20Techniques.md#91-サンプリングと-temperature再訪)
- [9.2　top-k / top-p の直感](Chapter%209%20Inference-Time%20Techniques.md#92-top-k-top-p-の直感)
- [9.3　KV キャッシュで速くする（直感）](Chapter%209%20Inference-Time%20Techniques.md#93-kv-キャッシュで速くする直感)
- [9.4　本章のまとめ](Chapter%209%20Inference-Time%20Techniques.md#94-本章のまとめ)

## [第10章　文脈長とスケーリングの直感](Chapter%2010%20Context%20Length%20and%20Scaling.md)

- [10.1　長い文脈をどう扱うか](Chapter%2010%20Context%20Length%20and%20Scaling.md#101-長い文脈をどう扱うか)
- [10.2　Attention の計算量という制約（第6巻8章の回収）](Chapter%2010%20Context%20Length%20and%20Scaling.md#102-attention-の計算量という制約第6巻8章の回収)
- [10.3　長文脈化の工夫の方向性（概観）](Chapter%2010%20Context%20Length%20and%20Scaling.md#103-長文脈化の工夫の方向性概観)
- [10.4　スケーリングとの関係](Chapter%2010%20Context%20Length%20and%20Scaling.md#104-スケーリングとの関係)
- [10.5　本章のまとめ](Chapter%2010%20Context%20Length%20and%20Scaling.md#105-本章のまとめ)

## [第11章　いま君が触っている GPT の地図](Chapter%2011%20A%20Map%20of%20the%20GPT%20You%20Use.md)

- [11.1　入力から出力までを全部追う](Chapter%2011%20A%20Map%20of%20the%20GPT%20You%20Use.md#111-入力から出力までを全部追う)
- [11.2　どの部品が1〜7巻のどこに対応するか](Chapter%2011%20A%20Map%20of%20the%20GPT%20You%20Use.md#112-どの部品が1〜7巻のどこに対応するか)
- [11.3　ブラックボックスだったものが地図になった](Chapter%2011%20A%20Map%20of%20the%20GPT%20You%20Use.md#113-ブラックボックスだったものが地図になった)
- [11.4　本章のまとめ](Chapter%2011%20A%20Map%20of%20the%20GPT%20You%20Use.md#114-本章のまとめ)

## [第12章　まとめ：結局、全部要る](Chapter%2012%20Summary%20Everything%20Is%20What%20You%20Need.md)

- [12.1　"○○ Is All You Need" ミームの回収](Chapter%2012%20Summary%20Everything%20Is%20What%20You%20Need.md#121-○○-is-all-you-need-ミームの回収)
- [12.2　どれも単独では all you need ではなかった](Chapter%2012%20Summary%20Everything%20Is%20What%20You%20Need.md#122-どれも単独では-all-you-need-ではなかった)
- [12.3　T Is All You Need の本当の意味](Chapter%2012%20Summary%20Everything%20Is%20What%20You%20Need.md#123-t-is-all-you-need-の本当の意味)
- [12.4　ここから先、各自がどこを深掘りするか](Chapter%2012%20Summary%20Everything%20Is%20What%20You%20Need.md#124-ここから先各自がどこを深掘りするか)
- [12.5　本章のまとめ](Chapter%2012%20Summary%20Everything%20Is%20What%20You%20Need.md#125-本章のまとめ)
