from typing import List, Dict, Optional, Tuple
import numpy as np
from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.bm25_retriever import BM25Retriever
from oil_rag.retrieval.reranker import BilingualReranker


class HybridRetriever:
    def __init__(self,
                 embedder: DocumentEmbedder,
                 indexer: FAISSIndexer,
                 bm25_retriever: Optional[BM25Retriever] = None,
                 reranker: Optional[BilingualReranker] = None,
                 fusion_method: str = "rrf",
                 alpha: float = 0.5,
                 dense_k: int = 200,
                 bm25_k: int = 200,
                 final_k: int = 20):
        self.embedder = embedder
        self.indexer = indexer
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
    
    def retrieve(self, 
                 query: str, 
                 k: Optional[int] = None,
                 use_bm25: bool = True,
                 use_reranker: bool = True) -> Tuple[List[Dict], List[float]]:
        k = k if k is not None else self.final_k
        if use_bm25 and self.bm25_retriever:
            documents, scores = self._hybrid_retrieve(query)
        else:
            documents, scores = self._dense_only_retrieve(query)
        if use_reranker and self.reranker and len(documents) > 0:
            rerank_input_k = min(len(documents), max(k * 2, 50))
            documents = self.reranker.rerank(query, documents[:rerank_input_k], top_k=k)
            scores = [doc.get("rerank_score", 0.0) for doc in documents]
        else:
            documents = documents[:k]
            scores = scores[:k]
        return documents, scores
    
    def _dense_only_retrieve(self, query: str) -> Tuple[List[Dict], List[float]]:
        query_embedding = self.embedder.embed_texts(query)
        results = self.indexer.search(query_embedding, k=self.dense_k)
        documents = [r["metadata"] for r in results]
        scores = [r["score"] for r in results]
        return documents, scores
    
    def _hybrid_retrieve(self, query: str) -> Tuple[List[Dict], List[float]]:
        bm25_results = self.bm25_retriever.retrieve(query, top_k=self.bm25_k)
        dense_docs, dense_scores = self._dense_only_retrieve(query)
        if self.fusion_method == "weighted":
            fused = self._weighted_fusion(bm25_results, dense_docs, dense_scores)
        elif self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(bm25_results, dense_docs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        documents = [item["doc"] for item in fused]
        scores = [item["score"] for item in fused]
        return documents, scores
    
    def _weighted_fusion(self, 
                        bm25_results: List[Dict],
                        dense_docs: List[Dict],
                        dense_scores: List[float]) -> List[Dict]:
        bm25_score_map = {}
        for doc in bm25_results:
            doc_id = doc.get("id", "")
            bm25_score_map[doc_id] = doc.get("bm25_score", 0.0)
        dense_score_map = {}
        for doc, score in zip(dense_docs, dense_scores):
            doc_id = doc.get("id", "")
            dense_score_map[doc_id] = score
        bm25_max = max(bm25_score_map.values()) if bm25_score_map else 1.0
        dense_max = max(dense_score_map.values()) if dense_score_map else 1.0
        all_doc_ids = set(bm25_score_map.keys()) | set(dense_score_map.keys())
        doc_map = {}
        for doc in bm25_results + dense_docs:
            doc_id = doc.get("id", "")
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        fused = []
        for doc_id in all_doc_ids:
            bm25_norm = bm25_score_map.get(doc_id, 0.0) / bm25_max if bm25_max > 0 else 0.0
            dense_norm = dense_score_map.get(doc_id, 0.0) / dense_max if dense_max > 0 else 0.0
            final_score = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm
            fused.append({
                "doc": doc_map[doc_id],
                "score": final_score
            })
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused
    
    def _reciprocal_rank_fusion(self,
                               bm25_results: List[Dict],
                               dense_docs: List[Dict],
                               k: int = 60) -> List[Dict]:
        scores = {}
        doc_map = {}
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.get("id", "")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc
        for rank, doc in enumerate(dense_docs, 1):
            doc_id = doc.get("id", "")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        fused = [
            {"doc": doc_map[doc_id], "score": score}
            for doc_id, score in scores.items()
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused
