"""
Configuration loader for NeuroMove Enterprise.
Loads config.yaml with environment variable substitution.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    _instance: Optional['Config'] = None
    _data: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._data:
            self.load()
    
    def load(self, path: Optional[str] = None) -> None:
        """Load config from YAML file with env var substitution."""
        if path is None:
            path = os.environ.get("NEUROMOVE_CONFIG", 
                str(Path(__file__).parent.parent / "config.yaml"))
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Substitute environment variables
        for key, value in os.environ.items():
            content = content.replace(f"${{{key}}}", value)
            content = content.replace(f"${key}", value)
        
        self._data = yaml.safe_load(content)
        
        # Ensure paths are Path objects
        if 'paths' in self._data:
            for k, v in self._data['paths'].items():
                if isinstance(v, str):
                    self._data['paths'][k] = Path(v).expanduser().resolve()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'minirocket.num_kernels')."""
        keys = key.split('.')
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# Global config instance
config = Config()


def get_config() -> Config:
    """Get global config instance."""
    return config


if __name__ == "__main__":
    cfg = get_config()
    print(f"Project root: {cfg['paths.project_root']}")
    print(f"MiniRocket kernels: {cfg['minirocket.num_kernels']}")
    print(f"Sampling freq: {cfg['preprocessing.sfreq']}")