from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from speeder.preprocessing.feature import FeatureFactory
from speeder.model.trainer import Trainer

@hydra.main(config_path="config/baseline.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    feature_factory = FeatureFactory(cfg, cv)
    feature_factory.create()

    trainer = Trainer(cfg, cv)
    trainer.run_train_cv()

if __name__ == "__main__":
    main()
