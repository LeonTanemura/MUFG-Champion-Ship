import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .utils import feature_name_combiner

logger = logging.getLogger(__name__)


# Copied from https://github.com/pfnet-research/deep-table.
# Modified by somaonishi and shoyameguro.
class TabularDataFrame(object):
    columns = [
        "",
        "",
    ]
    continuous_columns = []
    categorical_columns = []
    binary_columns = []
    selected_columns = []
    target_column = "score"

    def __init__(
        self,
        seed,
        categorical_encoder="ordinal",
        continuous_encoder: str = None,
        **kwargs,
    ) -> None:
        """
        Args:
            root (str): Path to the root of datasets for saving/loading.
            download (bool): If True, you must implement `self.download` method
                in the child class. Defaults to False.
        """
        self.seed = seed
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder

        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))
        self.id = self.test["Unnamed: 0"]
        
        self.label_encoder = LabelEncoder().fit(self.train[self.target_column])
        # self.train[self.target_column] = self.label_encoder.transform(self.train[self.target_column])


class V0(TabularDataFrame):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train = pd.read_csv(to_absolute_path("datasets/train_fix7.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test_fix7.csv"))
        # self.selected_columns = list(filter(lambda x: x not in ['Unnamed: 0','review', 'score', 'replyContent', 'new_review', 'new_replyContent', 'thumbsUpCount', 'reviewCreatedVersion', 'hoursToReply'], self.train.columns))
        self.selected_columns = list(filter(lambda x: x not in ['Unnamed: 0','review', 'score', 'replyContent', 'new_review', 'new_replyContent'], self.train.columns))
        # self.selected_columns = list(filter(lambda x: 'use' not in x and 
        #                                        'robertembedder' not in x and 
        #                                        'bertembedder' not in x and 
        #                                        'tfid' not in x and 
        #                                        'cntvec' not in x and 
        #                                        x not in ['Unnamed: 0', 'review', 'score', 'replyContent', 'new_review', 'new_replyContent'], 
        #                             self.train.columns))

        print(self.train[self.selected_columns].info())

class V1(TabularDataFrame):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train = pd.read_csv(to_absolute_path("datasets/train_fix15.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test_fix15.csv"))
        print(self.train.info())
        self.selected_columns = list(filter(lambda x: x not in ['Unnamed: 0','review', 'score', 'replyContent'], self.train.columns))

# 最終的に使用したもの
class V2(TabularDataFrame):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train = pd.read_csv(to_absolute_path("datasets/train_fix_final.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test_fix_final.csv"))
        print(self.train.info())
        # 使用する特徴量
        self.selected_columns = list(filter(lambda x: x not in ['Unnamed: 0','review', 'score', 'replyContent'], self.train.columns))