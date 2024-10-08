# MUFG Data Science Champion Ship 2024
Signateのコンペティションで1位に入賞したときのコードです。

https://signate.jp/competitions/1413

## 環境設定
- python 3.10.6

poetryで管理してるので，以下でパッケージをインストールする．
```bash
poetry install
```

## 実行
```bash
python make_features.py
```
・make_features.pyを実行し、tfidfとcountvecの特徴量を含んだcsvファイルを作成する。

・mufg-stacking.ipynbをkagglenotebookで実行し、bert, deberta, robertaの予測値を含んだcsvファイルを作成する。
```bash
python preprocess.py
```
・preprocess.pyを実行し、作成したcsvファイルをtrainデータとtestデータに組み合わせる。'hoursToReply’の特徴量も作成する。
・特徴量を追加したcsvファイルとモデルの学習に使用する特徴量をdataset/dataset.pyで指定する。
```bash
python main.py
```
