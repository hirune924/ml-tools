from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

@hydra.main(config_path="../config/baseline.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    ## Load Data
    target = 'Survived'
    train_X = pd.read_csv(utils.to_absolute_path(cfg.data.train_csv_path))
    train_Y = train_X[target]
    train_X = train_X.drop(target, axis=1)
    test_X = pd.read_csv(utils.to_absolute_path(cfg.data.test_csv_path))

    print(train_X.head())

    ## Feature Extraction

    ## Training

    ## Inference

if __name__ == "__main__":
    main()
