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

import seaborn as sns

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

    # Predict for test
    dtest = xgb.DMatrix(test_df)
    preds = []
    feature_importances = pd.DataFrame()
    for fold in range(len(val_scores)):
        # For predict
        model_path = os.path.join('model', f'xgboost_fold{fold}.model')
        model = Data.load(model_path)
        pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
        preds.append(pred)
        # For Feature importance
        fold_importance_df = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient='index', columns=['importance'])
        feature_importances = pd.concat([feature_importances, fold_importance_df], axis=1, join='outer')

    # save result of test predict
    pred_avg = np.mean(preds, axis=0)
    Data.dump(pred_avg, f'pred/test.pkl')

    # save result of feature importance
    fi_avg = feature_importances.mean(axis='columns').sort_values(ascending=False)
    sort_index = fi_avg.index
    os.makedirs(os.path.dirname(f'importance/feature_importance.csv'), exist_ok=True)
    feature_importances.reindex(index=sort_index).to_csv(f'importance/feature_importance.csv')
    experiment.log_metrics(dic=fi_avg.to_dict())

    # For Submission
    sub_df['Survived'] = pred_avg
    os.makedirs(os.path.dirname(f'submission/xgb_submission.csv'), exist_ok=True)
    sub_df.to_csv(f'submission/xgb_submission.csv')

if __name__ == "__main__":
    main()
