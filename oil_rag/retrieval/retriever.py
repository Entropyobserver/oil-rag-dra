from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch


class BilingualRetriever:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        top_k: int = 20,
        similarity_threshold: float = 0.7
    ):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.device = device
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.index = {}
        self.chunks = {}

    def add_documents(self, documents: List[Dict], language: str):
        if language not in self.index:
            self.index[language] = []
            self.chunks[language] = []
        
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        self.index[language] = embeddings
        self.chunks[language] = documents

    def retrieve(
        self,
        query: str,
        language: str = "en",
        top_k: Optional[int] = None
    ) -> List[Dict]:
        if language not in self.index:
            return []
        
        k = top_k or self.top_k
        
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarities = util.cos_sim(
            query_embedding,
            self.index[language]
        )[0].numpy()
        
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.similarity_threshold:
                chunk = self.chunks[language][idx].copy()
                chunk["score"] = score
                results.append(chunk)
        
        return results

    def retrieve_bilingual(
        self,
        query: str,
        primary_language: str = "en",
        secondary_language: str = "no",
        top_k: int = 10
    ) -> Dict[str, List[Dict]]:
        primary_results = self.retrieve(query, primary_language, top_k)
        secondary_results = self.retrieve(query, secondary_language, top_k)
        
        return {
            primary_language: primary_results,
            secondary_language: secondary_results
        }
