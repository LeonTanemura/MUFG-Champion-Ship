import logging
import os
from copy import deepcopy
from statistics import mean

import optuna
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

from model import get_classifier, get_regressor

import optuna.visualization as ov
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger(__name__)


def xgboost_config(trial: optuna.Trial, model_config, name=""):
    model_config.max_depth = trial.suggest_int("max_depth", 3, 10)
    model_config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    model_config.colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    model_config.reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
    model_config.reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
    model_config.n_estimators = trial.suggest_int("n_estimators", 100, 10000)
    return model_config

def lightgbm_config(trial: optuna.Trial, model_config, name=""):
    model_config.max_depth = trial.suggest_int("max_depth", 3, 10)
    model_config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    model_config.num_leaves = trial.suggest_int("num_leaves", 2, 256, log=True)
    model_config.colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    model_config.reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
    model_config.reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
    model_config.n_estimators = trial.suggest_int("n_estimators", 100, 10000)
    return model_config

def xgblgbm_config(trial: optuna.Trial, model_config, name=""):
    # XGBoostのパラメータ設定
    model_config.xgboost.max_depth = trial.suggest_int("xgboost_max_depth", 3, 10)
    model_config.xgboost.learning_rate = trial.suggest_float("xgboost_learning_rate", 1e-4, 1.0, log=True)
    model_config.xgboost.colsample_bytree = trial.suggest_float("xgboost_colsample_bytree", 0.1, 1.0)
    model_config.xgboost.reg_alpha = trial.suggest_float("xgboost_reg_alpha", 1e-8, 1.0, log=True)
    model_config.xgboost.reg_lambda = trial.suggest_float("xgboost_reg_lambda", 1e-8, 1.0, log=True)
    model_config.xgboost.n_estimators = trial.suggest_int("xgboost_n_estimators", 100, 10000)
    
    # LightGBMのパラメータ設定
    model_config.lightgbm.max_depth = trial.suggest_int("lightgbm_max_depth", 3, 10)
    model_config.lightgbm.learning_rate = trial.suggest_float("lightgbm_learning_rate", 1e-4, 1.0, log=True)
    model_config.lightgbm.num_leaves = trial.suggest_int("lightgbm_num_leaves", 2, 256, log=True)
    model_config.lightgbm.colsample_bytree = trial.suggest_float("lightgbm_colsample_bytree", 0.1, 1.0)
    model_config.lightgbm.reg_alpha = trial.suggest_float("lightgbm_reg_alpha", 1e-8, 1.0, log=True)
    model_config.lightgbm.reg_lambda = trial.suggest_float("lightgbm_reg_lambda", 1e-8, 1.0, log=True)
    model_config.lightgbm.n_estimators = trial.suggest_int("lightgbm_n_estimators", 100, 10000)

    return model_config

def get_model_config(model_name):
    if model_name == "xgboost":
        return xgboost_config
    elif model_name == "lightgbm":
        return lightgbm_config
    elif model_name == "xgblgbm":
        return xgblgbm_config
    else:
        raise ValueError()

def update_model_config(default_config, best_config):
    for _p, v in best_config.items():
        # モデル名を抽出
        if _p.startswith("xgboost_"):
            model_name = "xgboost"
            param_name = _p[len("xgboost_"):]
        elif _p.startswith("lightgbm_"):
            model_name = "lightgbm"
            param_name = _p[len("lightgbm_"):]
        else:
            # モデル名のプレフィックスがない場合、そのままキーを使用
            model_name = None
            param_name = _p

        # default_configの該当モデルにアクセス
        if model_name is None:
            # モデル名が指定されていない場合（単独で回す場合）
            if param_name in default_config:
                default_config[param_name] = v
        elif model_name in default_config:
            # モデル名が指定されている場合
            default_config[model_name][param_name] = v

    return default_config


class OptimParam:
    def __init__(
        self,
        model_name,
        default_config,
        input_dim,
        output_dim,
        X,
        y,
        val_data,
        columns,
        target_column,
        n_trials,
        n_startup_trials,
        storage,
        study_name,
        cv=True,
        n_jobs=1,
        seed=42,
        alpha=1,
        task="None",
    ) -> None:
        self.model_name = model_name
        self.default_config = deepcopy(default_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = get_model_config(model_name)
        self.X = X
        self.y = y
        self.val_data = val_data
        self.columns = columns
        self.target_column = target_column
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.storage = to_absolute_path(storage) if storage is not None else None
        self.study_name = study_name
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.alpha = alpha
        self.task = task

    def fit(self, model_config, X_train, y_train, X_val=None, y_val=None):
        if X_val is None and y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()
        
        model = get_regressor(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            seed=self.seed,
        )
        model.fit(
            X_train,
            y_train,
            # eval_set=(X_val, y_val),
            eval_set=[(X_train, y_train), (X_val, y_val)],
        )
        score = model.evaluate(
            self.val_data[self.columns],
            self.val_data[self.target_column].values.squeeze(),
        )
        return score

    def cross_validation(self, model_config):
        skf = StratifiedKFold(n_splits=10, random_state=self.seed, shuffle=True)
        ave = []
        for _, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, y_train = self.X.iloc[train_idx], self.y[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]
            score = self.fit(model_config, X_train, y_train, X_val, y_val)
            ave.append(score["QWK"])
        return mean(ave)

    def one_shot(self, model_config):
        score = self.fit(model_config, self.X, self.y)
        return score["QWK"]

    def objective(self, trial):
        _model_config = self.model_config(trial, deepcopy(self.default_config))
        if self.cv:
            value = self.cross_validation(_model_config)
        else:
            value = self.one_shot(_model_config)
        return value

    def get_n_complete(self, study: optuna.Study):
        n_complete = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
        return n_complete
    
    def get_best_config(self):
        if self.storage is not None:
            os.makedirs(self.storage, exist_ok=True)
            self.storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{self.storage}/optuna.db",
            )
        study = optuna.create_study(
            storage=self.storage,
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=self.n_startup_trials,
            ),
            load_if_exists=True,
        )
        n_complete = self.get_n_complete(study)
        n_trials = self.n_trials
        if n_complete > 0:
            n_trials -= n_complete
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)
        update_model_config(self.default_config, study.best_params)
        return self.default_config
