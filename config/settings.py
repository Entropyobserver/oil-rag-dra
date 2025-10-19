"""基本配置设置"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" 
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"

# 模型配置
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
DEFAULT_LLM_MODEL = "google/mt5-base"

# DRA配置
DRA_CONFIG = {
    "r_min": 4,
    "r_max": 32,
    "hidden_dims": [32, 16],
    "dropout": 0.1
}

# 检索配置  
RETRIEVAL_CONFIG = {
    "top_k": 20,
    "similarity_threshold": 0.7,
    "max_iterations": 3
}
