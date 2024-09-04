import numpy as np

from ..base_model import BaseRegressor
from ..gbm import LightGBMRegressor, XGBoostRegressor


class XGBLGBMRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostRegressor(input_dim, output_dim, model_config=self.model_config.xgboost, verbose=0, seed=seed)
        self.lgbm_model = LightGBMRegressor(input_dim, output_dim, model_config=self.model_config.lightgbm, verbose=0, seed=seed)

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)

    def predict(self, X):
        n = 0.386
        return (self.xgb_model.predict(X)*n + self.lgbm_model.predict(X)*(1-n))
        # return self.lgbm_model.predict(X)

    def predict_proba(self, X):
        return (self.xgb_model.predict_proba(X) + self.lgbm_model.predict_proba(X)) / 2

    def feature_importance(self):
        return self.xgb_model.feature_importance(), self.lgbm_model.feature_importance()
