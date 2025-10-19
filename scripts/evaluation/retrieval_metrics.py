from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict


class RetrievalMetrics:
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], 
                    relevant_ids: List[str], 
                    k: int) -> float:
        
        if not relevant_ids:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], 
                       relevant_ids: List[str], 
                       k: int) -> float:
        
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_k & relevant_set)
        return hits / k
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], 
                            relevant_ids: List[str]) -> float:
        
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], 
                  relevant_ids: List[str], 
                  k: int) -> float:
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def compute_all_metrics(retrieved_ids: List[str], 
                           relevant_ids: List[str], 
                           k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        
        metrics = {}
        
        for k in k_values:
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k)
        
        metrics["mrr"] = RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        
        return metrics


class RetrievalEvaluator:
    
    def __init__(self, test_queries: List[Dict]):
        self.test_queries = test_queries
    
    def evaluate_retriever(self, retriever, k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict:
        
        all_metrics = defaultdict(list)
        
        for query_data in self.test_queries:
            query = query_data["query"]
            relevant_ids = query_data["relevant_doc_ids"]
            
            documents, scores = retriever.retrieve(query, k=max(k_values))
            retrieved_ids = [doc.get("id", "") for doc in documents]
            
            metrics = RetrievalMetrics.compute_all_metrics(
                retrieved_ids, 
                relevant_ids, 
                k_values
            )
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        avg_metrics = {
            key: np.mean(values) 
            for key, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def compare_retrievers(self, retrievers: Dict[str, any]) -> Dict:
        
        results = {}
        
        for name, retriever in retrievers.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate_retriever(retriever)
        
        return results
    
    def print_comparison(self, results: Dict):
        
        print("\n" + "="*80)
        print("RETRIEVAL COMPARISON RESULTS")
        print("="*80)
        
        metrics_order = ["recall@1", "recall@5", "recall@10", "precision@5", "mrr", "ndcg@10"]
        
        retriever_names = list(results.keys())
        
        print(f"\n{'Metric':<15}", end="")
        for name in retriever_names:
            print(f"{name:<15}", end="")
        print()
        print("-"*80)
        
        for metric in metrics_order:
            print(f"{metric:<15}", end="")
            for name in retriever_names:
                value = results[name].get(metric, 0.0)
                print(f"{value:<15.4f}", end="")
            print()