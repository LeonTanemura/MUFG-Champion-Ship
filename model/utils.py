import numpy as np
from sklearn.metrics import f1_score, log_loss, cohen_kappa_score
import lightgbm as lgb
import xgboost as xgb


def f1_micro(y_true, y_pred):
    return -f1_score(y_true, y_pred, average="micro", zero_division=0)


def f1_micro_lgb(y_true, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return "f1_micro", f1_score(y_true, y_pred, average="micro", zero_division=0), True

def binary_logloss(y_true, y_pred):
    return "binary_logloss", log_loss(y_true, y_pred), False

def quadratic_weighted_kappa(y_true, y_pred):
    a = 0
    if isinstance(y_pred, xgb.QuantileDMatrix):
        # XGB
        y_true, y_pred = y_pred, y_true

        y_true = (y_true.get_label() + a).round()
        y_pred = (y_pred + a).clip(0, 4).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk

    else:
        # For lgb
        y_true = y_true + a
        y_pred = (y_pred + a).clip(0, 4).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk, True

def qwk_obj(y_true, y_pred):
    a = 0
    b = 1
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(0, 4)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess

# a = 2.948
# b = 1.092

