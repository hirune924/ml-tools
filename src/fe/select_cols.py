import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from speeder.feature.feature_manager import cached_feature
import pandas as pd
from load_data import load_data

cols_definition = {
    'target': ['Survived'],
    'numerical': ['Age', 'SibSp', 'Parch', 'Fare'],
    'categorical': ['Pclass', 'Sex'],
    'ignore': ['PassengerId', 'Name', 'Ticket']
}

@cached_feature(feature_name='select_cols', directory='features', ignore_columns = None)
def select_cols(train_csv, test_csv, target_col=None, test=True):
    train_df, test_df = load_data('input/train.csv', 'input/test.csv', test=True)

    return train_df[cols_definition[target_col]], test_df[cols_definition[target_col]]

select_cols('input/train.csv', 'input/test.csv', target_col='numerical', test=True)