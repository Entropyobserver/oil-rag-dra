from typing import List, Dict
from sentence_transformers import CrossEncoder
import torch


class BilingualReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = CrossEncoder(model_name, device=device)
        self.device = device

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        if not documents:
            return []
        
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        reranked = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        return reranked[:top_k]