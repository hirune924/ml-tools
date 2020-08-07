from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from speeder.preprocessing.feature import FeatureFactory

@hydra.main(config_path="config/baseline.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    feature_factory = FeatureFactory(cfg, cv)
    feature_factory.create()


if __name__ == "__main__":
    main()
