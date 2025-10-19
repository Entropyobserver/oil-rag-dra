import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class VectorIndex:
    def __init__(self, dimension: int, use_gpu=False):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        self.use_gpu = use_gpu
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata length mismatch")
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, top_k=20):
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        return results
    
    def search_with_filter(self, query_embedding: np.ndarray, 
                          filters: Dict, top_k=20, max_candidates=100):
        scores, indices = self.index.search(query_embedding, max_candidates)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            if self._matches_filters(meta, filters):
                results.append({
                    'score': float(score),
                    'metadata': meta
                })
                
            if len(results) >= top_k:
                break
        
        return results
    
    def _matches_filters(self, metadata: Dict, filters: Dict):
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            meta_value = metadata[key]
            
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif isinstance(value, dict):
                if 'min' in value and meta_value < value['min']:
                    return False
                if 'max' in value and meta_value > value['max']:
                    return False
            else:
                if meta_value != value:
                    return False
        
        return True
    
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path / 'index.faiss'))
        else:
            faiss.write_index(self.index, str(path / 'index.faiss'))
        
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str):
        path = Path(path)
        
        self.index = faiss.read_index(str(path / 'index.faiss'))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        with open(path / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
    
    @property
    def size(self):
        return self.index.ntotal


class BilingualIndex:
    def __init__(self, dimension: int, use_gpu=False):
        self.en_index = VectorIndex(dimension, use_gpu)
        self.no_index = VectorIndex(dimension, use_gpu)
        self.dimension = dimension
    
    def add_english(self, embeddings: np.ndarray, metadata: List[Dict]):
        self.en_index.add(embeddings, metadata)
    
    def add_norwegian(self, embeddings: np.ndarray, metadata: List[Dict]):
        self.no_index.add(embeddings, metadata)
    
    def search(self, query_embedding: np.ndarray, language: str, 
               top_k=20, filters=None):
        index = self.en_index if language == 'en' else self.no_index
        
        if filters:
            return index.search_with_filter(query_embedding, filters, top_k)
        else:
            return index.search(query_embedding, top_k)
    
    def search_both(self, query_embedding: np.ndarray, top_k=20):
        en_results = self.en_index.search(query_embedding, top_k)
        no_results = self.no_index.search(query_embedding, top_k)
        
        for r in en_results:
            r['language'] = 'en'
        for r in no_results:
            r['language'] = 'no'
        
        combined = en_results + no_results
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        return combined[:top_k]
    
    def save(self, path: str):
        path = Path(path)
        self.en_index.save(path / 'en')
        self.no_index.save(path / 'no')
    
    def load(self, path: str):
        path = Path(path)
        self.en_index.load(path / 'en')
        self.no_index.load(path / 'no')