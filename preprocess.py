import pandas as pd
import numpy as np
import statistics as st
import re

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train2 = pd.read_csv("datasets/train_stacking_deberta_review_v2.csv")
test2 = pd.read_csv("datasets/test_stacking_deberta_review_v2.csv")
train3 = pd.read_csv("datasets/train_stacking_deberta_replyContent_v2.csv")
test3 = pd.read_csv("datasets/test_stacking_deberta_replyContent_v2.csv")
train4 = pd.read_csv("datasets/train_stacking_bert_review_v2.csv")
test4 = pd.read_csv("datasets/test_stacking_bert_review_v2.csv")
train5 = pd.read_csv("datasets/train_stacking_bert_replyContent_v2.csv")
test5 = pd.read_csv("datasets/test_stacking_bert_replyContent_v2.csv")
train6 = pd.read_csv("datasets/train_stacking_roberta_review_v2.csv")
test6 = pd.read_csv("datasets/test_stacking_roberta_review_v2.csv")
train7 = pd.read_csv("datasets/train_stacking_roberta_replyContent_v2.csv")
test7 = pd.read_csv("datasets/test_stacking_roberta_replyContent_v2.csv")

train['deberta_review_pred'] = train2['deberta_review_pred']
test['deberta_review_pred'] = test2['deberta_review_pred']
train['deberta_replyContent_pred'] = train3['deberta_replyContent_pred']
test['deberta_replyContent_pred'] = test3['deberta_replyContent_pred']

train['bert_review_pred'] = train4['bert_review_pred']
test['bert_review_pred'] = test4['bert_review_pred']
train['bert_replyContent_pred'] = train5['bert_replyContent_pred']
test['bert_replyContent_pred'] = test5['bert_replyContent_pred']

train['roberta_review_pred'] = train6['roberta_review_pred']
test['roberta_review_pred'] = test6['roberta_review_pred']
train['roberta_replyContent_pred'] = train7['roberta_replyContent_pred']
test['roberta_replyContent_pred'] = test7['roberta_replyContent_pred']

train8 = pd.read_csv("datasets/new_train_tfid_features.csv")
test8 = pd.read_csv("datasets/new_test_tfid_features.csv")
train9 = pd.read_csv("datasets/new_train_cntvec_features.csv")
test9 = pd.read_csv("datasets/new_test_cntvec_features.csv")

train = pd.concat([train, train8], axis=1)
train = pd.concat([train, train9], axis=1)
test = pd.concat([test, test8], axis=1)
test = pd.concat([test, test9], axis=1)


train_test = pd.concat([train, test])

# 欠損値の確認
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

def convert_time_string_to_hours(time_str):
    pattern = r'(\d+)d(\d+)h'
    match = re.match(pattern, time_str)
    if match:
        days, hours = map(int, match.groups())
        total_hours = days * 24 + hours
        return total_hours
    return 0

# 分、秒の情報の削除
df['RetimeToReply'] = df['timeToReply'].astype(str).apply(convert_time_string)
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
train.to_csv('datasets/train_fix_final.csv', index=False)
test.to_csv('datasets/test_fix_final.csv', index=False)
