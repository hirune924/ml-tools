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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from speeder.feature.feature_manager import load_features, load_feature   
from speeder.utils import flatten_dict_cfg, Data

import xgboost as xgb

from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score, mean_squared_error, average_precision_score, accuracy_score


@hydra.main(config_path="../config/modeling_xgb.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    # For comet_ml
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
    sub_df = load_feature('_test.ftr', dir=utils.to_absolute_path('features'), ignore_columns = None)[['PassengerId']]

    # Train CV
    val_scores = []
    val_idxes = []
    val_preds = []
    for fold, (train_index, valid_index) in enumerate(cv.split(train_df, target_df)):
        dtrain = xgb.DMatrix(train_df.iloc[train_index], label=target_df.iloc[train_index])
        dvalid = xgb.DMatrix(train_df.iloc[valid_index], label=target_df.iloc[valid_index])
        params = cfg.model.params
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        # train
        model = xgb.train(params=dict(params), dtrain=dtrain, num_boost_round=cfg.model.num_boost_round, evals=watchlist,
                            obj=None, feval=None, maximize=False, early_stopping_rounds=cfg.model.early_stopping_rounds,
                            evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None)
        # predict for validation
        val_pred = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
        # calc metric
        val_score = accuracy_score(target_df.iloc[valid_index], np.where(val_pred>0.5, 1, 0))
        # save model
        model_path = os.path.join('model', f'xgboost_fold{fold}.model')
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Data.dump(model, model_path)
        # save logs
        val_idxes.append(valid_index)
        val_scores.append(val_score)
        val_preds.append(val_pred)
    # aggrigate results
    val_idxes = np.concatenate(val_idxes)
    order = np.argsort(val_idxes)
    val_preds = np.concatenate(val_preds, axis=0)
    val_preds = val_preds[order]
    # calc cv_score
    cv_score = accuracy_score(target_df, np.where(val_preds>0.5, 1, 0))

    Data.dump(val_preds, f'pred/train_oof.pkl')
    log_dict = {}
    log_dict['cv_score'] = cv_score
    for i in range(len(val_scores)):
        log_dict[f'fold_{i}_score'] = val_scores[i]
    experiment.log_metrics(dic=log_dict)


    #trainer = Trainer(configs=cfg, X_train=train_df, y_train=target_df, X_test=test_df, cv=cv, experiment=experiment)
    #trainer.run_train_cv()
    #trainer.run_predict_cv()
    #trainer.submission(sub_df)

if __name__ == "__main__":
    main()
