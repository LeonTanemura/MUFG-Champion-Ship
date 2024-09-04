from experiment.utils import set_seed

from .gbm import LightGBMClassifier, XGBoostClassifier, LightGBMRegressor, XGBoostRegressor, CatBoostRegressor
from .ensemble import XGBLGBMRegressor


def get_classifier(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, verbose, seed)
    else:
        raise KeyError(f"{name} is not defined.")

def get_regressor(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "catboost":
        return CatBoostRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "xgblgbm":
        return XGBLGBMRegressor(input_dim, output_dim, model_config, verbose, seed)
    else:
        raise KeyError(f"{name} is not defined.")
