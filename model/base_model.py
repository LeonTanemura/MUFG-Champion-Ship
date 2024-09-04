from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error, 
    mean_absolute_error,
    cohen_kappa_score,
)
import numpy as np
import pandas as pd


class BaseClassifier:
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        self.verbose = verbose

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict_proba(self, X):
        return self.model.predict_proba(X.values)

    def predict(self, X):
        return self.model.predict(X.values)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        results = {}
        results["ACC"] = accuracy_score(y, y_pred)
        y_score = self.predict_proba(X)
        y_score = y_score[:, 1]
        results["AUC"] = roc_auc_score(y, y_score)
        results["Precision"] = precision_score(y, y_pred, average="micro", zero_division=0)
        results["Recall"] = recall_score(y, y_pred, average="micro", zero_division=0)
        results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="micro", zero_division=0)
        results["F1"] = f1_score(y, y_pred, average="micro", zero_division=0)
        return results

    def feature_importance(self):
        raise NotImplementedError()

class BaseRegressor:
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        self.verbose = verbose

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict(self, X):
        return self.model.predict(X.values)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        results = {}
        mse = mean_squared_error(y, y_pred)
        results["MSE"] = mse
        results["MAE"] = mean_absolute_error(y, y_pred)
        results["RMSE"] = np.sqrt(mse)
        # y_pred_rounded = np.round(y_pred).astype(int)
        # y_rounded = np.round(y).astype(int)
        # results["QWK"] = cohen_kappa_score(y_rounded, y_pred_rounded, weights='quadratic')
        # 回帰モデルの予測値を整数ラベルに変換
        thresholds=[0.65, 1.5, 2.5, 3.5]
        pred_labels = pd.cut(y_pred, bins=[-np.inf] + thresholds + [np.inf], 
                             labels=[0, 1, 2, 3, 4]).astype(int)
        
        true_labels = y.astype(int)
        results["QWK"] = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')
        return results

    def feature_importance(self):
        raise NotImplementedError()


