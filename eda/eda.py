from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="../config/config.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

if __name__ == "__main__":
    main()
