# BERT-nfr-classification-jp

## インストール方法  
- NVIDIA CUDA toolkit 11.3 以上
- Python 3.10.6 以上  

以上2点が導入されている必要があります．  
以下のコマンドを実行してください．  
```
python -m venv venv
```
```
venv/Scripts/activate.ps1
```
```
pip install -r requirements.txt
```

## ファイルについて
- ``nfr_classifier_multi.py`` : マルチラベル分類を行うプログラムです．研究の適用評価で使用したプログラムです．
- ``nfr_classifier_multi-v2.py``: 研究のSHAPによる可視化においてモデル作成に使ったプログラムです． 
- ``nfr_classifer_multi_shap.ipynb``: 研究のSHAPによる可視化の結果表示に使ったプログラムです．
- ``nfr_classifier.py``: シングルラベル分類を行うプログラムです．研究では使用していません．
- ``nfr_classifier_shap.ipynb``: シングルラベル分類のモデルでSHAPの可視化を行うプログラムです．研究では使用していません．
- ``Google-Colab/nfr_classifier_multi-colab.ipynb``: Google Colaboratoryで実行するためのファイルです．

BERTの事前学習モデルを変更するには ``MODEL_NAME`` の値を編集してください．  
学習データを変更するには ``TRAIN_PATH`` の値を編集してください．  
同様にテストデータを変える場合は ``TEST_PATH`` の値を編集してください．  
``FOLD`` : 検証データと学習データの分割数  
``LABEL_LIST`` : 学習，分類の対象に入れるラベル  
``THRESHOLD`` : マルチラベル分類の推論時に用いる閾値 デフォルトは0です．  

## 実行オプション
- ``--skip_training`` : ファインチューニングのプロセスをスキップして推論のみ行います．
- ``--gen`` : (experimental) FOLD回ファインチューニングとモデル生成を行い交差検証を行います．研究では使っていません．実行完了まで時間がかかります．"nfr_classifier_multi-v2.py"のみで使えるオプションです．
- ``--convert_model`` : 拡張子が.ckptのモデルをpytorchが直接読み込める形式に変換します．例: ``--convert_model path/to/ckpt `` "nfr_classifier_multi-v2.py"のみで使えるオプションです．
- ``--merge_checkpoint`` : (experimental) model_ml/以下にあるチェックポイントの重みを平均化したモデルを生成します．正しく動かないかもしれません．"nfr_classifier_multi-v2.py"のみで使えるオプションです．

## 実行結果について  

プログラムを実行すると以下のファイルが生成されます．
- classification_result_ml.xlsx : デバッグ用のファイルです．分類結果と正解ラベル，スコアの詳細などが書かれたエクセルシートです．
- classification_analysys_ml.xlsx : 非機能要求レポートのエクセルシートです．定量要約の結果などが含まれます．
- model_ml/XXX.ckpt : pytorch lightningライブラリで読み込めるファインチューニング済みモデルです．
- model_transformers_ml/ : (v2のみ)pytorchライブラリで読み込めるファインチューニング済みモデルです．

## issue
- nfr_classifier_multi_shap.ipynbでSHAPの可視化結果を全て表示しようとすると，ファイルが破損してしまうバグがあります．現状では数十件の結果を表示後にCtrl+Cで実行を中断してください．
- v2と適用評価で用いたプログラムでは精度に違いが生じます．v2では閾値``THRESHOLD``の調整が必要となります．