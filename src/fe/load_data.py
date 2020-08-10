import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from speeder.feature.feature_manager import cached_feature
import pandas as pd

@cached_feature(feature_name='', directory='features', ignore_columns = None)
def load_data(train_csv, test_csv, test=True):
    return pd.read_csv(train_csv), pd.read_csv(test_csv)

load_data('input/train.csv', 'input/test.csv', test=True)