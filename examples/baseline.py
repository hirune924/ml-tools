import os

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

import numpy as np
import pandas as pd

import xfeat

@hydra.main(config_path="../config/baseline.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    ## Load Data
    feature_name = "features/train_test.ftr"
    if not os.path.exists(utils.to_absolute_path(feature_name)):
        target = cfg.data.target_col_name
        train_X = pd.read_csv(utils.to_absolute_path(cfg.data.train_csv_path))
        test_X = pd.read_csv(utils.to_absolute_path(cfg.data.test_csv_path))

        xfeat.utils.compress_df(pd.concat([
            train_X, test_X,
            ], sort=False)).reset_index(drop=True).to_feather(utils.to_absolute_path("features/train_test.ftr"))

    print(pd.read_feather(utils.to_absolute_path("features/train_test.ftr")).head())

    ## Feature Extraction
    feature_name = "features/feature_num_features.ftr"
    if not os.path.exists(utils.to_absolute_path(feature_name)):
        print("Save numerical features")
        xfeat.SelectNumerical().fit_transform(
            pd.read_feather(utils.to_absolute_path("features/train_test.ftr"))
        ).reset_index(drop=True).to_feather(utils.to_absolute_path(feature_name))
    print(pd.read_feather(utils.to_absolute_path(feature_name)).head())

    feature_name = "features/feature_1way_label_encoding.ftr"
    if not os.path.exists(utils.to_absolute_path(feature_name)):
        print("Categorical encoding using label encoding")
        xfeat.Pipeline([
            xfeat.SelectCategorical(), xfeat.LabelEncoder(output_suffix="")]
        ).fit_transform(
            pd.read_feather(utils.to_absolute_path("features/train_test.ftr"))
        ).reset_index(drop=True).to_feather(utils.to_absolute_path(feature_name))
    print(pd.read_feather(utils.to_absolute_path(feature_name)).head())

    feature_name = "features/feature_2way_label_encoding.ftr"
    if not os.path.exists(utils.to_absolute_path(feature_name)):
        print("2-order combination of categorical features")
        xfeat.Pipeline([
        xfeat.SelectCategorical(),
        xfeat.ConcatCombination(drop_origin=True, r=2),
        xfeat.LabelEncoder(output_suffix=""),
        ]).fit_transform(
            pd.read_feather(utils.to_absolute_path("features/train_test.ftr"))
        ).reset_index(drop=True).to_feather(utils.to_absolute_path(feature_name))
    print(pd.read_feather(utils.to_absolute_path(feature_name)).head())

    feature_name = "features/feature_3way__including_Sex_label_encoding.ftr"
    if not os.path.exists(utils.to_absolute_path(feature_name)):
        print("3-order combination of categorical features")
        xfeat.Pipeline([
        xfeat.SelectCategorical(),
        xfeat.ConcatCombination(drop_origin=True, include_cols=["Sex"], r=3),
        xfeat.LabelEncoder(output_suffix=""),
        ]).fit_transform(
            pd.read_feather(utils.to_absolute_path("features/train_test.ftr"))
        ).reset_index(drop=True).to_feather(utils.to_absolute_path(feature_name))
    print(pd.read_feather(utils.to_absolute_path(feature_name)).head())


    ## Training

    ## Inference

if __name__ == "__main__":
    main()
