from pathlib import Path
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.data = self._load_config("data.yaml")
        self.model = self._load_config("model.yaml")

    def _load_config(self, filename: str) -> Dict[str, Any]:
        config_path = self.config_dir / filename
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config {filename}: {e}")
            return {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        config_section = getattr(self, section, {})
        return config_section.get(key, default)

    def get_nested(self, *keys, default: Any = None) -> Any:
        value = self.__dict__
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value


config = ConfigManager()