import pandas as pd
import numpy as np
import statistics as st
import re


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

def convert_time_string(time_str):
    if isinstance(time_str, str):
        pattern = r'(\d+)\s+days\s+(\d+):.*'
        match = re.match(pattern, time_str)
        if match:
            days, hours = match.groups()
            return f"{days}d{hours}h"
    return "Invalid format"

df['RetimeToReply'] = df['timeToReply'].astype(str).apply(convert_time_string)

def convert_time_string_to_hours(time_str):
    pattern = r'(\d+)d(\d+)h'
    match = re.match(pattern, time_str)
    if match:
        days, hours = map(int, match.groups())
        total_hours = days * 24 + hours
        return total_hours
    return 0

# timeToReply列を時間に変換して新しい列に追加
df['hoursToReply'] = df['RetimeToReply'].astype(str).apply(convert_time_string_to_hours)

# 補完するターゲットの設定
targets = ['reviewCreatedVersion']
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
targets = ['timeToReply', 'RetimeToReply']
df = df.drop(targets, axis=1)

missing_value_checker(df, "train_test")

train_test = df

# trainとtestに再分割
train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]
test = test.drop('score', axis=1)

print(train.info())
print(test.info())

# csvファイルの作成
train.to_csv('datasets/train_fix.csv', index=False)
test.to_csv('datasets/test_fix.csv', index=False)

# targets = ['review', 'replyContent']
# train = train.drop(targets, axis=1)
# train.to_csv('datasets/train_drop_str.csv', index=False)
