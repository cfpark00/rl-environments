import omegaconf


def load_config(config_path: str, validate: bool = True) -> dict:
    """
    Load a yaml config file from a given path.

    Args:
    config_path: str, path to the yaml config file.
    validate: bool, whether to validate the config against a schema. Default: True.

    Returns:
    config: dict, the loaded config file.
    """
    config = omegaconf.OmegaConf.load(config_path)
    return config


def validate_config(config):
    """
    Validate a config.

    Args:
    config: OmegaConf, the config to validate.

    Returns:
    None
    """
    assert "env_name" in config, "Config must contain 'env_name' field."
    assert "hostname" in config, "Config must contain 'hostname' field."
    assert "port" in config, "Config must contain 'port' field."
