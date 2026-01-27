import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    path : str
        Path to the YAML config file (relative or absolute)

    Returns
    -------
    dict
        Parsed configuration
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
