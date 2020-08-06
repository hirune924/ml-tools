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


@hydra.main(config_path="../config/eda.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    ## Load Data
    df = pd.read_csv(utils.to_absolute_path(cfg.eda.train_csv_path))
    print(df.head())

    if cfg.eda.pandas_profiling:
        ## pandas_profiling report
        profile = ProfileReport(df, title="Pandas Profiling Report")
        profile.to_file("pd-profile-report.html")

    if cfg.eda.sweetviz:
        ## SweetVis report
        # for single df
        my_report = sv.analyze(df, target_feat=None, feat_cfg=None, pairwise_analysis='auto')
        # for multi df
        # my_report = sv.compare([train_df, "Training Data"], [test_df, "Test Data"], target_feat="Survived")
        # for single df intra
        # my_report = sv.compare_intra(my_dataframe, my_dataframe["Sex"] == "male", ["Male", "Female"], feature_config)

        my_report.show_html("sweetvis-report.html")

if __name__ == "__main__":
    main()
