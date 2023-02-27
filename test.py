from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="config", config_name="agent_based")
def my_app(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    my_app()
