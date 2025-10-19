import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from config.config import config
from oil_rag.core.system_builder import get_retriever
from oil_rag.evaluation.retrieval_metrics import RetrievalEvaluator


def load_test_queries() -> list:
    test_path = config.paths.results_dir / "test_queries.json"
    
    if not test_path.exists():
        print(f"Test queries not found at {test_path}")
        print("Run scripts/create_test_queries.py first")
        return []
    
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    test_queries = load_test_queries()
    
    if not test_queries:
        return
    
    test_queries = [q for q in test_queries if q.get("relevant_doc_ids")]
    
    print(f"Loaded {len(test_queries)} annotated test queries")
    
    retrievers = {
        "BM25": get_retriever(mode="bm25", use_reranker=False),
        "Dense": get_retriever(mode="dense", use_reranker=False),
        "Hybrid": get_retriever(mode="hybrid", use_reranker=False),
        "Hybrid+Rerank": get_retriever(mode="hybrid", use_reranker=True)
    }
    
    evaluator = RetrievalEvaluator(test_queries)
    
    results = evaluator.compare_retrievers(retrievers)
    
    evaluator.print_comparison(results)
    
    output_path = config.paths.results_dir / "retrieval_evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_by_metric = {}
    for metric in results["BM25"].keys():
        best_name = max(results.keys(), key=lambda x: results[x][metric])
        best_value = results[best_name][metric]
        best_by_metric[metric] = (best_name, best_value)
    
    for metric, (name, value) in best_by_metric.items():
        print(f"{metric:<20} Best: {name:<15} ({value:.4f})")


if __name__ == "__main__":
    run_evaluation()
