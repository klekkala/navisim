"""
Configuration Module

Configuration management for NaviSim-Isaac integration.
"""

from typing import Dict, Any
import yaml
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    default_config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")
    if os.path.exists(default_config_path):
        return load_config(default_config_path)
    return {}
