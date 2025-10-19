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
        
        documents, scores = retriever.retrieve(test_query, k=5)
        
        print(f"Retrieved {len(documents)} documents")
        print(f"Top score: {scores[0]:.4f}")
        print(f"Top document: {documents[0].get('text', '')[:100]}...")
        print(f"Year: {documents[0].get('year', 'N/A')}")


def test_reranker_effect():
    test_query = "safety measures and environmental protection"
    
    print("\n" + "="*60)
    print("RERANKER EFFECT COMPARISON")
    print("="*60)
    
    retriever_no_rerank = get_retriever(mode="hybrid", use_reranker=False)
    retriever_with_rerank = get_retriever(mode="hybrid", use_reranker=True)
    
    print("\n--- WITHOUT RERANKER ---")
    docs_no, scores_no = retriever_no_rerank.retrieve(test_query, k=3)
    for i, (doc, score) in enumerate(zip(docs_no, scores_no)):
        print(f"{i+1}. Score: {score:.4f} | {doc.get('text', '')[:80]}...")
    
    print("\n--- WITH RERANKER ---")
    docs_yes, scores_yes = retriever_with_rerank.retrieve(test_query, k=3)
    for i, (doc, score) in enumerate(zip(docs_yes, scores_yes)):
        print(f"{i+1}. Score: {score:.4f} | {doc.get('text', '')[:80]}...")


if __name__ == "__main__":
    print(f"Using index: {config.paths.faiss_index}")
    print(f"Using documents: {config.paths.documents_pkl}")
    
    test_retrieval_modes()
    test_reranker_effect()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)