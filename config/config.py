from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def results_dir(self) -> Path:
        return self.project_root / "evaluation_results"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    @property
    def processed_data(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def aligned_data(self) -> Path:
        return self.processed_data / "aligned"
    
    @property
    def faiss_index(self) -> Path:
        return self.models_dir / "faiss_index.bin"
    
    @property
    def documents_pkl(self) -> Path:
        return self.models_dir / "documents.pkl"
    
    def ensure_dirs(self):
        for path in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_model: str = "google/mt5-base"
    device: str = field(default_factory=lambda: "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu")
    embedding_dim: int = 768
    max_seq_length: int = 512


@dataclass
class RetrievalConfig:
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    dense_top_k: int = 200
    bm25_top_k: int = 200
    fusion_method: str = "rrf"
    fusion_alpha: float = 0.5
    rerank_top_k: int = 50
    final_top_k: int = 20
    similarity_threshold: float = 0.5


@dataclass
class DRAConfig:
    r_min: int = 4
    r_max: int = 32
    hidden_dims: list = field(default_factory=lambda: [32, 16])
    dropout: float = 0.1
    input_dim: int = 8


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dra: DRAConfig = field(default_factory=DRAConfig)
    
    def __post_init__(self):
        self.paths.ensure_dirs()


config = AppConfig()


def get_config() -> AppConfig:
    return config


def update_config(**kwargs):
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
