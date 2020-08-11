from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from speeder.model.trainer import Trainer

@hydra.main(config_path="../config/modeling.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

    trainer = Trainer(cv)
    trainer.run_train_cv()

if __name__ == "__main__":
    main()
