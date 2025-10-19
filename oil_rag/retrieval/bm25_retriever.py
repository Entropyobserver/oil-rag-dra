from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re
from pathlib import Path


class BM25Retriever:
    def __init__(self, 
                 documents: List[Dict],
                 k1: float = 1.5,
                 b: float = 0.75,
                 language: str = "en"):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.language = language
        
        self.tokenized_corpus = [
            self._tokenize(doc.get("text", "")) 
            for doc in documents
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)
        
        self.doc_id_to_idx = {
            doc.get("id", f"doc_{i}"): i 
            for i, doc in enumerate(documents)
        }
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        tokenized_query = self._tokenize(query)
        
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 20) -> List[List[Dict]]:
        return [self.retrieve(q, top_k) for q in queries]