# import comet_ml in the top of your file
from comet_ml import Experiment
    
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="QCxbRVX2qhQj1t0ajIZl2nk2c",
                        project_name="ml-tools", workspace="hirune924",
                        auto_param_logging=False)

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from speeder.model.trainer import Trainer
from speeder.feature.feature_manager import load_features, load_feature   
from speeder.utils import flatten_dict_cfg
@hydra.main(config_path="../config/modeling.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    experiment_name = '/'.join(os.getcwd().split('/')[-2:])
    experiment.set_name(experiment_name)
    experiment.log_parameters(flatten_dict_cfg(cfg, sep='/'))
    # Define CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

    # Load Features
    feature_list = cfg['feature']
    feature_names = [f + '_train.ftr' for f in feature_list]
    train_df = load_features(feature_names, dir=utils.to_absolute_path('features'), ignore_columns = None)
    feature_names = [f + '_test.ftr' for f in feature_list]
    test_df = load_features(feature_names, dir=utils.to_absolute_path('features'), ignore_columns = None)
    target_df = load_feature('_train.ftr', dir=utils.to_absolute_path('features'), ignore_columns = None)[['Survived']]

    #print(train_df.head())
    #print(test_df.head())
    #print(target_df.head())

    trainer = Trainer(configs=cfg, X_train=train_df, y_train=target_df, X_test=test_df, cv=cv, experiment=experiment)
    trainer.run_train_cv()

if __name__ == "__main__":
    main()
