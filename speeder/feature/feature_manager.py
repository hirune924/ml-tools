import os
import pandas as pd

import pyarrow
import functools
import warnings
from tqdm import tqdm

def save_feature(train_df=None, test_df=None, save_dir=None, feature_name=None):
    feature_path = os.path.join(save_dir, feature_name)
    if train_df is not None:
        train_df.to_feather(feature_path + '_train.ftr')
    if test_df is not None:
        test_df.to_feather(feature_path + '_test.ftr')

def load_feature(feature_file_name, dir, ignore_columns=None):
    feature_path = os.path.join(dir, feature_file_name)

    df = pd.read_feather(feature_path)
    if ignore_columns:
        return df.drop([c for c in ignore_columns if c in df.columns], axis=1)
    else:
        return df

def concat_df(base_df, add_df, duplicate_suffix=None):
    base_columns = list(base_df.columns)
    add_columns = list(add_df.columns)

    if len(base_df) != len(add_df):
        warnings.warn('Length of base_df is {}, but length of add_df is {}.'.format(str(len(base_df)), str(len(add_df))))
    for c in add_columns:
        if c in base_columns:
            warnings.warn('A feature name {} is duplicated.'.format(c))
            if duplicate_suffix is None:
                add_df = add_df.drop(c, axis=1)
                warnings.warn('The duplicated feature {} in add_df is dropped.'.format(c))
            else:
                rename = c 
                while rename in base_columns:
                    rename += '_' + duplicate_suffix
                add_df = add_df.rename(columns={c: rename})
                warnings.warn('The duplicated name in feature={} will be renamed to {}'.format(c, rename))
    return pd.concat([base_df, add_df], axis=1)

def load_features(feature_names, dir = None, ignore_columns = None):

    dfs = [load_feature(f, dir=dir, ignore_columns=ignore_columns) for f in tqdm(feature_names)]

    concatenated = dfs[0]
    for idx, df in enumerate(dfs[1:]):
        concat_df(concatenated, df, duplicate_suffix=feature_names[idx + 1].rstrip('_train.ftr').rstrip('_test.ftr'))

    return concatenated

def cached_feature(feature_name, directory, ignore_columns = None):

    def _decorator(fun):
        @functools.wraps(fun)
        def _decorated_fun(*args, **kwargs):
            options = '_' + '_'.join([i + '=' + str(kwargs[i]) for i in kwargs if i != 'test'])
            f_name = feature_name + options
            try:
                if not kwargs['test']:
                    return load_feature(f_name + '_train.ftr', directory, ignore_columns), None
                else:
                    return load_feature(f_name + '_train.ftr', directory, ignore_columns), load_feature(f_name + '_test.ftr', directory, ignore_columns)
            except (pyarrow.ArrowIOError, IOError):
                train_df, test_df = fun(*args, **kwargs)
                save_feature(train_df, test_df, directory, f_name)
                return train_df, test_df

        return _decorated_fun

    return _decorator