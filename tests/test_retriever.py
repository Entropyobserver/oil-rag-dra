import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import config
from oil_rag.core.system_builder import get_retriever


def test_retrieval_modes():
    test_query = "What was oil production in 2020?"
    
    print("="*60)
    print("RETRIEVAL MODE COMPARISON")
    print("="*60)
    
    modes = ["bm25", "dense", "hybrid"]
    
    for mode in modes:
        print(f"\n--- {mode.upper()} MODE ---")
        
        retriever = get_retriever(mode=mode, use_reranker=(mode=="hybrid"))
        documents, scores = retriever.retrieve(test_query, k=20)
        
        print(f"Requested: 20 documents")
        print(f"Returned: {len(documents)}")
        print(f"Top score: {scores[0]:.4f}")
        print(f"Top doc: {documents[0].get('text', '')[:100]}...")
        print(f"Year: {documents[0].get('year', 'N/A')}")
        
        if len(documents) >= 10:
            print(f"10th score: {scores[9]:.4f}")
        if len(documents) >= 20:
            print(f"20th score: {scores[19]:.4f}")


def test_reranker_effect():
    test_query = "safety measures and environmental protection"
    
    print("\n" + "="*60)
    print("RERANKER EFFECT COMPARISON")
    print("="*60)
    
    retriever_no_rerank = get_retriever(mode="hybrid", use_reranker=False)
    docs_no, scores_no = retriever_no_rerank.retrieve(test_query, k=10)
    
    print("\n--- WITHOUT RERANKER ---")
    print(f"Documents returned: {len(docs_no)}")
    for i, (doc, score) in enumerate(zip(docs_no[:3], scores_no[:3]), 1):
        print(f"{i}. Score:{score:.4f} | {doc.get('text', '')[:80]}...")
    
    retriever_with_rerank = get_retriever(mode="hybrid", use_reranker=True)
    docs_yes, scores_yes = retriever_with_rerank.retrieve(test_query, k=10)
    
    print("\n--- WITH RERANKER ---")
    print(f"Documents returned: {len(docs_yes)}")
    for i, (doc, score) in enumerate(zip(docs_yes[:3], scores_yes[:3]), 1):
        print(f"{i}. Score:{score:.4f} | {doc.get('text', '')[:80]}...")


def test_config_values():
    print("\n" + "="*60)
    print("CURRENT CONFIG VALUES")
    print("="*60)
    print(f"dense_top_k: {config.retrieval.dense_top_k}")
    print(f"bm25_top_k: {config.retrieval.bm25_top_k}")
    print(f"final_top_k: {config.retrieval.final_top_k}")
    print(f"fusion_method: {config.retrieval.fusion_method}")
    print(f"similarity_threshold: {config.retrieval.similarity_threshold}")
    print(f"rerank_top_k: {config.retrieval.rerank_top_k}")
    print(f"bm25_k1: {config.retrieval.bm25_k1}")
    print(f"bm25_b: {config.retrieval.bm25_b}")


if __name__ == "__main__":
    print(f"Using FAISS index: {config.paths.faiss_index}")
    print(f"Using documents: {config.paths.documents_pkl}")
    print()
    
    test_config_values()
    test_retrieval_modes()
    test_reranker_effect()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
