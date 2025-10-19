import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from config.config import config


class FAISSIndexer:
    def __init__(self, 
                 dimension: int = 768,
                 index_type: str = "Flat",
                 use_gpu: bool = False):
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.documents = []
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
    
    def add(self, embeddings: np.ndarray, documents: List[Dict]):
        if embeddings.shape[0] != len(documents):
            raise ValueError("Embeddings and documents length mismatch")
        
        faiss.normalize_L2(embeddings)
        
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 10) -> List[Dict]:
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and idx != -1:
                results.append({
                    "score": float(score),
                    "metadata": self.documents[idx]
                })
        
        return results
    
    def save(self, 
             index_path: Optional[Path] = None,
             documents_path: Optional[Path] = None):
        
        index_path = index_path or config.paths.faiss_index
        documents_path = documents_path or config.paths.documents_pkl
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        with open(documents_path, "wb") as f:
            pickle.dump(self.documents, f)
    
    def load(self, 
             index_path: Optional[Path] = None,
             documents_path: Optional[Path] = None):
        
        index_path = index_path or config.paths.faiss_index
        documents_path = documents_path or config.paths.documents_pkl
        
        self.index = faiss.read_index(str(index_path))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)
    
    def get_index_size(self) -> int:
        return self.index.ntotal