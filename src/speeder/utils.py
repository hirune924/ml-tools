import os
import joblib

from omegaconf import DictConfig, OmegaConf
import pandas as pd

class Data:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

def flatten_dict_cfg(cfg, sep='.'):
    return pd.json_normalize(OmegaConf.to_container(cfg, resolve=True), sep=sep).to_dict(orient='records')[0]