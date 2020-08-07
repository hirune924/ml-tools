import os
import pandas as pd
import numpy as np
from hydra import utils
import itertools
from sklearn import preprocessing

class FeatureFactory:

    def __init__(self, configs: dict, cv=None):
        self.run_name = configs['fe_name']
        self.data = configs.data
        self.coldef = self.data.cols_definition
        self.fe = configs.fe
        self.cv = cv

    def create(self):
        print('Load data')

        for f in self.fe:
            print(f)
            utils.instantiate(f)


def load_data(train_csv, test_csv):
    print('Load Data')
    feature_name = "features/train_test.ftr"
    feature_abs_path = utils.to_absolute_path(feature_name)
    if not os.path.exists(feature_abs_path):
        train_df = pd.read_csv(utils.to_absolute_path(train_csv))
        test_df = pd.read_csv(utils.to_absolute_path(test_csv))

        pd.concat([
            train_df, test_df,
            ], sort=False).reset_index(drop=True).to_feather(feature_abs_path)

    print(pd.read_feather(feature_abs_path).head())


def numeric_interact_2order(target_col, input_feature):
    print('Numeric Interact 2nd Order')
    df = pd.read_feather(utils.to_absolute_path(input_feature))
    org_cols = df.columns.values
    feature_name = "features/numeric_interact_2order.ftr"
    feature_abs_path = utils.to_absolute_path(feature_name)
    if not os.path.exists(feature_abs_path):
        for col1, col2 in list(itertools.combinations(target_col, 2)):
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_sub_{col2}'] = df[col1] - df[col2]
            try:
                df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
            except:
                print(f'{col1}_div_{col2}')
        df.drop(org_cols, axis=1).reset_index(drop=True).to_feather(feature_abs_path)
    print(pd.read_feather(feature_abs_path).head())


def label_encoding(target_col, input_feature):
    print('Label Encoding')
    df = pd.read_feather(utils.to_absolute_path(input_feature))
    org_cols = df.columns.values
    feature_name = "features/label_encoding.ftr"
    feature_abs_path = utils.to_absolute_path(feature_name)
    if not os.path.exists(feature_abs_path):
        for f in target_col:
            try:
                lbl = preprocessing.LabelEncoder()
                df[f'{f}_lbl_encoded'] = lbl.fit_transform(list(df[f].values))
            except:
                print(f)
        df.drop(org_cols, axis=1).reset_index(drop=True).to_feather(feature_abs_path)
    print(pd.read_feather(feature_abs_path).head())

