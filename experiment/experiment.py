import logging
from time import time

import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

import dataset.dataset as dataset
from dataset import TabularDataFrame
from model import get_classifier, get_regressor

from .optuna import OptimParam
from .utils import cal_metrics, cal_metrics_regression, load_json, set_seed, plot_confusion_matrix, concatenate_images

from collections import Counter
import pickle

logger = logging.getLogger(__name__)


class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name

        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(seed=config.seed, **self.data_config)
        self.train, self.test = dataframe.train, dataframe.test
        self.columns = dataframe.selected_columns
        self.target_column = dataframe.target_column
        self.label_encoder = dataframe.label_encoder

        self.input_dim = len(self.columns)
        self.output_dim = len(self.label_encoder.classes_)

        self.id = dataframe.id

        self.seed = config.seed
        self.init_writer()

        self.save_model = config.save_model
        self.save_predict = config.save_predict
        self.existing_models = config.existing_models

        self.a = 0,
        self.b = 0,

    def init_writer(self):
        metrics = [
            "fold",
            "QWK",
        ]
        self.writer = {m: [] for m in metrics}

    def add_results(self, i_fold, scores: dict, time):
        self.writer["fold"].append(i_fold)
        for m in self.writer.keys():
            if m == "fold":
                continue
            self.writer[m].append(scores[m])

    def each_fold(self, i_fold, train_data, val_data):
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data)
    
        model = get_regressor(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            verbose=self.exp_config.verbose,
            seed=self.seed,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=[(x, y), (val_data[self.columns], val_data[self.target_column].values.squeeze())],
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        score_all = []
        if self.existing_models:
            model_filename = "/home/leon/study/mydir/MUFG-Champion-Ship/outputs/single/main/V2/2024-08-31/15-52-41/lightgbm.pkl"
            with open(model_filename, 'rb') as f:
                ex_models1 = pickle.load(f)
            model_filename = "/home/leon/study/mydir/MUFG-Champion-Ship/outputs/single/main/V2/2024-09-01/16-24-53/xgboost.pkl"
            with open(model_filename, 'rb') as f:
                ex_models2 = pickle.load(f)
            ex_score_all = []
        models = []
        image_list = []
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):
            if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
                logger.info(f"Skip {i_fold + 1} fold. Already finished.")
                continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)
            models.append(model)

            score = cal_metrics_regression(model, val_data, self.columns, self.target_column)
            score.update(model.evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
            score_all.append(score)
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/MSE: {score['MSE']:.4f} | val/MAE: {score['MAE']:.4f} |"
                f" val/RMSE: {score['RMSE']:.4f} | val/QWK: {score['QWK']:.4f}"
            )
            self.add_results(i_fold, score, time)

            # コンフュージョンマトリックスを計算してグラフを表示
            x_val, y_val = self.get_x_y(val_data)
            img = plot_confusion_matrix(model, x_val, y_val, i_fold)
            image_list.append(img)

            if self.existing_models:
                ex_score = cal_metrics_regression(ex_models1[i_fold], val_data, self.columns, self.target_column)
                ex_score.update(ex_models1[i_fold].evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
                ex_score_all.append(ex_score)
                ex_score = cal_metrics_regression(ex_models2[i_fold], val_data, self.columns, self.target_column)
                ex_score.update(ex_models2[i_fold].evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
                ex_score_all.append(ex_score)

        concatenated_img = concatenate_images(image_list)
        if concatenated_img:
            concatenated_img.save('concatenated_confusion_matrices.png')

        final_score = Counter()
        for item in score_all:
            final_score.update(item)

        logger.info(
                f"[{self.model_name} results] MSE: {(final_score['MSE']/self.n_splits)} | MAE: {(final_score['MAE']/self.n_splits)} | "
                f"RMSE: {(final_score['RMSE']/self.n_splits)} | QWK: {(final_score['QWK']/self.n_splits)}"
            )

        if self.existing_models:
            for item in ex_score_all:
                final_score.update(item)
            self.n_splits = self.n_splits*3
            logger.info(
                    f"[ensemble results] MSE: {(final_score['MSE']/self.n_splits)} | MAE: {(final_score['MAE']/self.n_splits)} | "
                    f"RMSE: {(final_score['RMSE']/self.n_splits)} | QWK: {(final_score['QWK']/self.n_splits)}"
                )
            for model in ex_models1:
                models.append(model)
            for model in ex_models2:
                models.append(model)

        probabilities = []
        for model in models:
            proba= model.predict(self.test[self.columns])
            proba += self.a  # 必要に応じて調整
            print(proba)
            probabilities.append(proba)

        predictions = np.mean(probabilities, axis=0)
        logger.info(f"predictions: {predictions}")
        thresholds=[0.65, 1.5, 2.5, 3.5]
        predictions = pd.cut(predictions, [-np.inf] + thresholds + [np.inf], 
                    labels=[0,1,2,3,4]).astype('int32')

        if self.save_predict:
            pred = pd.DataFrame(predictions)
            pred.to_csv(f"{self.model_name}_pred.csv", index=False)

        if self.save_model:
            model_filename = f"{self.model_name}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(models, f)

        predictions = np.round(predictions.clip(0, 4)).astype(int)
        print(predictions)

        submit_df = pd.DataFrame(self.id)
        submit_df["score"] = predictions
        print(submit_df)
        print(self.train.columns)
        # self.train.to_csv("train_feature.csv", index=False)
        submit_df.to_csv("submission.csv", index=False, header=False)

    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def get_x_y(self, train_data):
        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y


class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        return self.model_config


class ExpOptuna(ExpBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_trials = config.exp.n_trials
        self.n_startup_trials = config.exp.n_startup_trials

        self.storage = config.exp.storage
        self.study_name = config.exp.study_name
        self.cv = config.exp.cv
        self.n_jobs = config.exp.n_jobs

    def run(self):
        if self.exp_config.delete_study:
            for i in range(self.n_splits):
                optuna.delete_study(
                    study_name=f"{self.exp_config.study_name}_{i}",
                    storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                )
                print(f"delete successful in {i}")
            return
        super().run()

    def get_model_config(self, i_fold, x, y, val_data):
        op = OptimParam(
            self.model_name,
            default_config=self.model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            X=x,
            y=y,
            val_data=val_data,
            columns=self.columns,
            target_column=self.target_column,
            n_trials=self.n_trials,
            n_startup_trials=self.n_startup_trials,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            cv=self.cv,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )
        return op.get_best_config()
