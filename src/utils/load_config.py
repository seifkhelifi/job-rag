import yaml
import os


def load_config(config_path="config.yml"):
    """
    Load configuration from a YAML file

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: Configuration data
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}
