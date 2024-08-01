import pandas as pd
import numpy as np
import statistics as st


train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])


def missing_value_checker(df, name):
    chk_null = df.isnull().sum()
    chk_null_pct = chk_null / (df.index.max() + 1)
    chk_null_tbl = pd.concat([chk_null[chk_null > 0], chk_null_pct[chk_null_pct > 0]], axis=1)
    chk_null_tbl = chk_null_tbl.rename(columns={0: "欠損数",1: "欠損割合"})
    print(name)
    print(chk_null_tbl, end="\n\n")

missing_value_checker(train_test, "train_test")
missing_value_checker(train, "train")
missing_value_checker(test, "test")

# 欠損値の補完(train)
df = train_test
print(df.info())

# 補完するターゲットの設定
targets = []
# mode(最頻値), mean(平均値), median(中央値)
val_name = "mode"

for target in targets:
    if val_name == "mode":
        value = st.mode(df[target])
    elif val_name == "mean":
        value = st.mean(df[target])
    elif val_name == "median":
        value = st.median(df[target])
    else:
        raise ValueError("Invalid value name. Please specify 'mode', 'mean', or 'median'.")
    
    # 欠損値を補完
    train_test[target] = train_test[target].fillna(value)

# 欠損値特徴量の削除
targets = []
df = df.drop(targets, axis=1)

train_test = df

# trainとtestに再分割
train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]
test = test.drop('', axis=1)

print(train.info())
print(test.info())

# csvファイルの作成
train.to_csv('datasets/train_fix.csv', index=False)
test.to_csv('datasets/test_fix.csv', index=False)
