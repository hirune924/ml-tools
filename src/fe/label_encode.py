import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from speeder.feature.feature_manager import cached_feature
import pandas as pd
from sklearn import preprocessing
from load_data import load_data

cols_definition = {
    'target': ['Survived'],
    'numerical': ['Age', 'SibSp', 'Parch', 'Fare'],
    'categorical': ['Pclass', 'Sex', 'Cabin', 'Embarked'],
    'ignore': ['PassengerId', 'Name', 'Ticket']
}

def label_encoding(train, test, target_cols):
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    org_cols = train.columns
    for f in cols_definition[target_cols]:
        try:
            lbl = preprocessing.LabelEncoder()
            train[f + '_label_encode'] = lbl.fit_transform(list(train[f].fillna('NaN').values))
        except:
            print(f)
    train = train.drop(org_cols, axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


@cached_feature(feature_name='label_encode', directory='features', ignore_columns = None)
def label_encode(train_csv, test_csv, target_col=None, test=True):
    train_df, test_df = load_data('input/train.csv', 'input/test.csv', test=True)
    train_df, test_df = label_encoding(train_df, test_df, target_cols=target_col)
    return train_df, test_df

label_encode('input/train.csv', 'input/test.csv', target_col='categorical', test=True)