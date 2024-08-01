from experiment.utils import set_seed

from .gbm import LightGBMClassifier, XGBoostClassifier, LightGBMRegressor, XGBoostRegressor
from .ensemble import XGBLGBMClassifier, XGB10Classifier, XGB7LGBM7Classifier, XGBLRClassifier


def get_classifier(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "xgblgbm":
        return XGBLGBMClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "xgb10":
        return XGB10Classifier(input_dim, output_dim, model_config, verbose)
    elif name == "xgb7lgbm7":
        return XGB7LGBM7Classifier(input_dim, output_dim, model_config, verbose)
    elif name == "xgblr":
        return XGBLRClassifier(input_dim, output_dim, model_config, verbose)
    else:
        raise KeyError(f"{name} is not defined.")

def get_regressor(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMRegressor(input_dim, output_dim, model_config, verbose, seed)
    else:
        raise KeyError(f"{name} is not defined.")
