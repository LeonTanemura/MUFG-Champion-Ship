# リポジトリ名
## 環境設定
- python 3.10.6

poetryで管理してるので，以下でパッケージをインストールする．
```bash
poetry install
```

## 実行
```bash
python main.py
```

## 手順
- リポジトリ作成からコミットまで
```bash
git clone URL
```
```bash
git add .
``` 
```bash
git commit -m "first commit"
```

- poetry作成
```bash
cd project_xyz
```
```bash
poetry init
```
```bash
poetry install
```
```bash
poetry add <package-name>
```
```bash
poetry add scikit-learn
poetry add hydra-core
poetry add optuna
poetry add xgboost
poetry add lightgbm
```
```bash
poetry update
```

- DATA

datasetsディレクトリを作成し、その中にサンプルデータを入れる。

preprocess.pyを使用し、データ処理

