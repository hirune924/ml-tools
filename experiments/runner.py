from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

import numpy as np
import pandas as pd

try:
    from pandas_profiling import ProfileReport
    # https://github.com/pandas-profiling/pandas-profiling
    _PANDASPROFILING_AVAILABLE = True
except ImportError:
    _PANDASPROFILING_AVAILABLE = False

try:
    import sweetviz as sv
    # https://github.com/fbdesignpro/sweetviz
    _SWEETVIZ_AVAILABLE = True
except ImportError:
    _SWEETVIZ_AVAILABLE = False


@hydra.main(config_path="config/baseline.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())



if __name__ == "__main__":
    main()
