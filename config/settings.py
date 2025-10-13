"""
配置管理模块
统一加载和管理项目配置
"""
from pathlib import Path
import yaml
from typing import Dict, Any, Optional


class Config:
    """项目配置管理器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / 'config'
        
        # 加载各类配置
        self.data = self._load_config('data.yaml')
        self.model = self._load_config('model.yaml')
        self.training = self._load_config('training.yaml')
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置值"""
        config_section = getattr(self, section, {})
        return config_section.get(key, default)
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """更新配置"""
        if hasattr(self, section):
            config_section = getattr(self, section)
            config_section.update(updates)


# 全局配置实例
config = Config()