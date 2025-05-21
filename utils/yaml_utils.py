# src/config_utils.py
import yaml
import os
from typing import Dict, Any, Optional

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_yaml_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Dictionary containing configuration parameters
        config_path: Path to save the YAML configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to {config_path}")

def merge_configs(base_config: Dict[str, Any], override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge a base configuration with an override configuration.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Optional override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    if override_config is None:
        return base_config.copy()
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def flatten_config(config: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten a nested configuration dictionary for logging purposes.
    
    Args:
        config: Configuration dictionary to flatten
        parent_key: Parent key prefix for nested items
        
    Returns:
        Flattened dictionary with dot notation keys
    """
    flattened = {}
    
    for k, v in config.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            flattened.update(flatten_config(v, new_key))
        else:
            flattened[new_key] = v
    
    return flattened

def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Load configuration from params.yaml
    """
    return load_yaml_config(config_path)