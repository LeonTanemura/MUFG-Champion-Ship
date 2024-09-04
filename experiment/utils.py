import json
import operator as op
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io

# グローバル閾値の定義
THRESHOLDS = [0.65, 1.5, 2.5, 3.5]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_json(data: Dict[str, Union[int, float, str]], save_dir: str = "./"):
    with open(os.path.join(save_dir, "results.json"), mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path) -> Dict[str, Union[int, float, str]]:
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_object(obj, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)

def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)

def cal_auc_score(model, data, feature_cols, label_col):
    pred_proba = model.predict_proba(data[feature_cols])
    if data[label_col].nunique() == 2:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba[:, 1])
    else:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba, multi_class="ovo")
    return auc

def cal_acc_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[label_col], pred)
    return acc

def cal_metrics(model, data, feature_cols, label_col):
    acc = cal_acc_score(model, data, feature_cols, label_col)
    auc = cal_auc_score(model, data, feature_cols, label_col)
    return {"ACC": acc, "AUC": auc}

def cal_mse_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    mse = mean_squared_error(data[label_col], pred)
    return mse

def cal_rmse_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    mse = mean_squared_error(data[label_col], pred)
    rmse = np.sqrt(mse)
    return rmse

def cal_mae_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    mae = mean_absolute_error(data[label_col], pred)
    return mae

def cal_qwk_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    pred_labels = pd.cut(pred, bins=[-np.inf] + THRESHOLDS + [np.inf], 
                         labels=[0, 1, 2, 3, 4]).astype('int32')
    
    true_labels = data[label_col].values.astype(int)
    
    # QWKの計算
    qwk = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')
    return qwk

def cal_metrics_regression(model, data, feature_cols, label_col):
    mse = cal_mse_score(model, data, feature_cols, label_col)
    mae = cal_mae_score(model, data, feature_cols, label_col)
    rmse = cal_rmse_score(model, data, feature_cols, label_col)
    qwk = cal_qwk_score(model, data, feature_cols, label_col)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "QWK": qwk}

def set_categories_in_rule(ruleset, categories_dict):
    ruleset.set_categories(categories_dict)

def plot_confusion_matrix(model, x_val, y_val, i_fold):
    # 予測を取得
    y_pred = model.predict(x_val)
    # 閾値に基づいて予測ラベルを変換
    pred_labels = pd.cut(y_pred, bins=[-np.inf] + THRESHOLDS + [np.inf], 
                         labels=[0, 1, 2, 3, 4]).astype('int32')
    # 実際のラベル
    true_labels = y_val.astype(int)
    # 混同行列の計算
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    # グラフの作成
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix for fold {i_fold+1}")
    # 画像をバッファに保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf)

def concatenate_images(image_list):
    # 画像リストが空の場合は処理を終了
    if not image_list:
        return None

    # 画像を読み込み、サイズを取得
    images = [img for img in image_list]
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)
    
    # 連結するためのキャンバスを作成
    concatenated_image = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    return concatenated_image
