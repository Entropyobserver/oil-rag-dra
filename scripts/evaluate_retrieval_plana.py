import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import List, Dict
from config.config import config
from oil_rag.core.system_builder import get_retriever


class PlanAEvaluator:
    
    def __init__(self, qa_jsonl_path: str):
        self.qa_dataset = []
        
        with open(qa_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line)
                self.qa_dataset.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'relevant_doc_ids': [qa['doc_id']],
                    'category': qa['category'],
                    'year': qa['year']
                })
        
        print(f"Loaded {len(self.qa_dataset)} questions")
    
    def recall_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)
    
    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if k == 0:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(retrieved_k & relevant_set)
        return hits / k
    
    def mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0
    
    def evaluate(self, mode: str = "hybrid", use_reranker: bool = True, k_values: List[int] = [1, 3, 5, 10, 20]):
        print(f"\n{'='*60}")
        print(f"Evaluation mode: {mode.upper()} | Reranker: {'✓' if use_reranker else '✗'}")
        print(f"{'='*60}")
        
        retriever = get_retriever(mode=mode, use_reranker=use_reranker)
        
        all_metrics = {f'recall@{k}': [] for k in k_values}
        all_metrics.update({f'precision@{k}': [] for k in k_values})
        all_metrics['mrr'] = []
        
        max_k = max(k_values)
        
        for i, qa in enumerate(self.qa_dataset, 1):
            question = qa['question']
            relevant_ids = qa['relevant_doc_ids']
            
            try:
                documents, scores = retriever.retrieve(question, k=max_k)
                retrieved_ids = [doc.get('id', '') for doc in documents]
                
                def convert_qa_to_index(qa_id):
                    if not qa_id:
                        return qa_id
                    parts = qa_id.split('_')
                    if len(parts) >= 4 and parts[2] == 'para':
                        return f"{parts[1]}_{parts[3]}_{parts[0]}"
                    return qa_id
                
                index_format_relevant = [convert_qa_to_index(rid) for rid in relevant_ids]
                    
            except Exception as e:
                print(f"Error retrieving for question {i}: {e}")
                retrieved_ids = []  # 如果检索失败，返回空列表
            
            for k in k_values:
                all_metrics[f'recall@{k}'].append(
                    self.recall_at_k(retrieved_ids, index_format_relevant, k)
                )
                all_metrics[f'precision@{k}'].append(
                    self.precision_at_k(retrieved_ids, index_format_relevant, k)
                )
            
            all_metrics['mrr'].append(
                self.mrr(retrieved_ids, index_format_relevant)
            )
            
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(self.qa_dataset)}")
        
        avg_metrics = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def print_results(self, metrics: Dict[str, float]):
        print(f"\n{'Metric':<15} {'Score':<10}")
        print("-"*30)
        
        for k in [1, 3, 5, 10, 20]:
            if f'recall@{k}' in metrics:
                print(f"{'Recall@'+str(k):<15} {metrics[f'recall@{k}']:.4f}")
        
        print()
        for k in [1, 3, 5, 10, 20]:
            if f'precision@{k}' in metrics:
                print(f"{'Precision@'+str(k):<15} {metrics[f'precision@{k}']:.4f}")
        
        print()
        print(f"{'MRR':<15} {metrics['mrr']:.4f}")
        print("="*60)
    
    def compare_modes(self):
        modes_config = [
            ("bm25", False),
            ("dense", False),
            ("hybrid", False),
            ("hybrid", True),
        ]
        
        results = {}
        
        for mode, use_reranker in modes_config:
            config_name = f"{mode}" + ("_rerank" if use_reranker else "")
            metrics = self.evaluate(mode, use_reranker)
            results[config_name] = metrics
        
        print(f"\n{'='*80}")
        print("Comparison of all configurations")
        print(f"{'='*80}")
        
        print(f"\n{'Config':<20}", end="")
        for k in [1, 5, 10, 20]:
            print(f"R@{k:<3}", end="  ")
        print("MRR")
        print("-"*80)
        
        for config_name, metrics in results.items():
            print(f"{config_name:<20}", end="")
            for k in [1, 5, 10, 20]:
                print(f"{metrics.get(f'recall@{k}', 0):.3f}", end="  ")
            print(f"{metrics.get('mrr', 0):.3f}")
        
        print("="*80)
        
        return results


def main():
    evaluator = PlanAEvaluator('/mnt/d/J/Desktop/language_technology/course/projects_AI/oil_rag_dra/data/test/test_qa.jsonl')
    
    results = evaluator.compare_modes()
    
    output_path = config.paths.results_dir / 'plan_a_evaluation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
