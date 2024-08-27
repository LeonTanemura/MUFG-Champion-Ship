import lightgbm as lgb
import xgboost as xgb
from lightgbm.callback import log_evaluation, early_stopping
from sklearn.utils.validation import check_X_y
import numpy as np
import pandas as pd

from .base_model import BaseClassifier, BaseRegressor
from .utils import f1_micro, f1_micro_lgb, binary_logloss, qwk_obj, quadratic_weighted_kappa


class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",  # 2値分類用のobjectiveに変更
            eval_metric="logloss",  # 2値分類なのでloglossを使用
            early_stopping_rounds=50,
            **self.model_config,
            random_state=seed
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(X, y, eval_set=eval_set, verbose=self.verbose > 0)
        # print("XGBoostモデルの使用されたパラメータ:", self.model.get_params())

    def feature_importance(self):
        return self.model.feature_importances_

class LightGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMClassifier(
            objective="binary",  # 2値分類用のobjectiveに変更
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=binary_logloss,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0)],
        )
        # print("LightGBMモデルの使用されたパラメータ:", self.model.get_params())
        
    def feature_importance(self):
        return self.model.feature_importances_


class XGBoostRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBRegressor(
            # objective="reg:squarederror",  # 目的関数を回帰用に変更
            objective=qwk_obj,
            **self.model_config,
            random_state=seed,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, y_val) for X_val, y_val in eval_set]

        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=10000),
            xgb.callback.EarlyStopping(100, metric_name="QWK", maximize=True, save_best=True)
        ]

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=quadratic_weighted_kappa,
            callbacks=xgb_callbacks,
            verbose=0
        )

    def feature_importance(self):
        return self.model.feature_importances_

class LightGBMRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMRegressor(
            objective=qwk_obj,
            # objective='regression',
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)

        eval_set = [(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, y_val) for X_val, y_val in eval_set]

        self.model.fit(
            X,
            y,
            eval_names=['train', 'valid'],
            eval_set=eval_set,
            eval_metric=quadratic_weighted_kappa, 
            callbacks=[early_stopping(stopping_rounds=100)],
            # callbacks=[log_evaluation(period=100), early_stopping(stopping_rounds=75)],
        )
        
    def feature_importance(self):
        return self.model.feature_importances_
