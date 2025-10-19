from pathlib import Path
from typing import Optional
import pickle

from config.config import config
from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.bm25_retriever import BM25Retriever
from oil_rag.retrieval.reranker import BilingualReranker
from oil_rag.retrieval.hybrid_retriever import HybridRetriever


class SystemBuilder:
    
    @staticmethod
    def build_embedder(device: Optional[str] = None) -> DocumentEmbedder:
        device = device or config.models.device
        return DocumentEmbedder(
            model_name=config.models.embedding_model,
            device=device
        )
    
    @staticmethod
    def build_indexer(use_gpu: bool = False) -> FAISSIndexer:
        return FAISSIndexer(
            dimension=config.models.embedding_dim,
            index_type="IVF",
            use_gpu=use_gpu
        )
    
    @staticmethod
    def load_indexer() -> FAISSIndexer:
        indexer = SystemBuilder.build_indexer()
        indexer.load()
        return indexer
    
    @staticmethod
    def build_bm25_retriever(documents: Optional[list] = None) -> BM25Retriever:
        if documents is None:
            documents = SystemBuilder._load_documents()
        
        return BM25Retriever(
            documents=documents,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b
        )
    
    @staticmethod
    def build_reranker(device: Optional[str] = None) -> BilingualReranker:
        device = device or config.models.device
        return BilingualReranker(
            model_name=config.models.reranker_model,
            device=device
        )
    
    @staticmethod
    def build_hybrid_retriever(
        use_bm25: bool = True,
        use_reranker: bool = True,
        device: Optional[str] = None
    ) -> HybridRetriever:
        
        embedder = SystemBuilder.build_embedder(device)
        indexer = SystemBuilder.load_indexer()
        
        bm25_retriever = None
        if use_bm25:
            documents = indexer.documents
            bm25_retriever = SystemBuilder.build_bm25_retriever(documents)
        
        reranker = None
        if use_reranker:
            reranker = SystemBuilder.build_reranker(device)
        
        return HybridRetriever(
            embedder=embedder,
            indexer=indexer,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            fusion_method=config.retrieval.fusion_method,
            alpha=config.retrieval.fusion_alpha,
            dense_k=config.retrieval.dense_top_k,
            bm25_k=config.retrieval.bm25_top_k,
            final_k=config.retrieval.final_top_k
        )
    
    @staticmethod
    def _load_documents() -> list:
        with open(config.paths.documents_pkl, "rb") as f:
            return pickle.load(f)


def get_retriever(
    mode: str = "hybrid",
    use_reranker: bool = True,
    device: Optional[str] = None
) -> HybridRetriever:
    
    if mode == "hybrid":
        return SystemBuilder.build_hybrid_retriever(
            use_bm25=True,
            use_reranker=use_reranker,
            device=device
        )
    elif mode == "dense":
        return SystemBuilder.build_hybrid_retriever(
            use_bm25=False,
            use_reranker=use_reranker,
            device=device
        )
    elif mode == "bm25":
        return SystemBuilder.build_hybrid_retriever(
            use_bm25=True,
            use_reranker=False,
            device=device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")