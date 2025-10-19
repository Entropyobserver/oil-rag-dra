#!/usr/bin/env python3
"""
简单检索评估脚本 - 方案A
每个问题只使用1个相关文档ID，计算基础检索指标
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.retrieval.retriever import HybridRetriever
    from src.models.embeddings import EmbeddingModel
    from config.settings import Config
except ImportError as e:
    print(f"警告: 无法导入模块 {e}")
    print("如果需要运行完整评估，请确保所有依赖都已安装")


class SimpleRetrievalEvaluator:
    """
    简单检索评估器 - 每个问题只有1个相关文档
    """
    
    def __init__(self, retriever=None):
        self.retriever = retriever
        
    def load_qa_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """加载QA数据集"""
        qa_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pairs.append(json.loads(line.strip()))
        return qa_pairs
    
    def calculate_metrics(self, results: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        计算检索指标
        
        Args:
            results: 每个结果包含 'relevant_docs' (set) 和 'retrieved_docs' (list)
            k_values: 计算Recall@K的K值列表
        """
        metrics = {}
        
        # 计算各种指标
        total_queries = len(results)
        
        # Recall@K
        for k in k_values:
            recall_at_k = 0
            for result in results:
                relevant_docs = result['relevant_docs']
                retrieved_docs_at_k = set(result['retrieved_docs'][:k])
                
                if len(relevant_docs) > 0:
                    recall = len(relevant_docs & retrieved_docs_at_k) / len(relevant_docs)
                    recall_at_k += recall
            
            metrics[f'Recall@{k}'] = recall_at_k / total_queries if total_queries > 0 else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for result in results:
            relevant_docs = result['relevant_docs']
            retrieved_docs = result['retrieved_docs']
            
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    mrr += 1.0 / (i + 1)
                    break
        
        metrics['MRR'] = mrr / total_queries if total_queries > 0 else 0.0
        
        # Hit Rate (至少检索到1个相关文档的查询比例)
        hits = 0
        for result in results:
            relevant_docs = result['relevant_docs']
            retrieved_docs = set(result['retrieved_docs'])
            if len(relevant_docs & retrieved_docs) > 0:
                hits += 1
        
        metrics['Hit_Rate'] = hits / total_queries if total_queries > 0 else 0.0
        
        return metrics
    
    def evaluate_with_mock_retriever(self, qa_dataset: List[Dict[str, Any]], 
                                   num_retrieved: int = 10) -> Dict[str, float]:
        """
        使用模拟检索器进行评估（用于演示）
        """
        results = []
        
        # 获取所有可能的文档ID
        all_doc_ids = list(set([qa['doc_id'] for qa in qa_dataset]))
        
        for qa in qa_dataset:
            query = qa['question']
            relevant_doc = qa['doc_id']
            
            # 模拟检索：随机选择文档，但确保相关文档有50%概率在前面
            import random
            random.seed(42)  # 固定随机种子
            
            retrieved_docs = []
            
            # 50%概率将相关文档放在前3位
            if random.random() < 0.5:
                position = random.randint(0, min(2, num_retrieved-1))
                other_docs = [doc for doc in all_doc_ids if doc != relevant_doc]
                random.shuffle(other_docs)
                
                for i in range(num_retrieved):
                    if i == position:
                        retrieved_docs.append(relevant_doc)
                    else:
                        if other_docs:
                            retrieved_docs.append(other_docs.pop())
            else:
                # 随机排序
                other_docs = [doc for doc in all_doc_ids if doc != relevant_doc]
                random.shuffle(other_docs)
                retrieved_docs = other_docs[:num_retrieved]
                
                # 25%概率完全没有相关文档
                if random.random() > 0.25 and len(retrieved_docs) > 0:
                    pos = random.randint(0, len(retrieved_docs)-1)
                    retrieved_docs[pos] = relevant_doc
            
            results.append({
                'query': query,
                'relevant_docs': {relevant_doc},
                'retrieved_docs': retrieved_docs[:num_retrieved]
            })
        
        return self.calculate_metrics(results)
    
    def evaluate_with_real_retriever(self, qa_dataset: List[Dict[str, Any]], 
                                   num_retrieved: int = 10) -> Dict[str, float]:
        """
        使用真实检索器进行评估
        """
        if self.retriever is None:
            raise ValueError("需要提供检索器才能进行真实评估")
        
        results = []
        
        for qa in qa_dataset:
            query = qa['question']
            relevant_doc = qa['doc_id']
            
            try:
                # 使用检索器检索文档
                retrieved_results = self.retriever.search(query, k=num_retrieved)
                retrieved_docs = [result['doc_id'] for result in retrieved_results]
                
                results.append({
                    'query': query,
                    'relevant_docs': {relevant_doc},
                    'retrieved_docs': retrieved_docs
                })
                
            except Exception as e:
                print(f"查询 '{query}' 检索失败: {e}")
                results.append({
                    'query': query,
                    'relevant_docs': {relevant_doc},
                    'retrieved_docs': []
                })
        
        return self.calculate_metrics(results)
    
    def print_detailed_results(self, metrics: Dict[str, float], 
                             qa_dataset: List[Dict[str, Any]] = None):
        """打印详细的评估结果"""
        print("=" * 60)
        print("检索评估结果 - 方案A (每个问题1个相关文档)")
        print("=" * 60)
        
        if qa_dataset:
            print(f"数据集大小: {len(qa_dataset)} 个查询")
            
            # 统计类别分布
            categories = {}
            for qa in qa_dataset:
                cat = qa.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            print("\n类别分布:")
            for cat, count in sorted(categories.items()):
                print(f"  {cat}: {count}")
        
        print("\n核心指标:")
        print("-" * 30)
        
        # 按重要性排序显示指标
        key_metrics = ['Recall@1', 'Recall@3', 'Recall@5', 'MRR', 'Hit_Rate']
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                print(f"{metric:12}: {value:.4f} ({value*100:.2f}%)")
        
        print("\n所有指标:")
        print("-" * 30)
        for metric, value in sorted(metrics.items()):
            print(f"{metric:15}: {value:.4f}")
        
        # 解释指标含义
        print("\n指标说明:")
        print("-" * 30)
        print("Recall@K    : 前K个结果中包含相关文档的查询比例")
        print("MRR         : 平均倒数排名，衡量相关文档的平均排名")
        print("Hit_Rate    : 至少检索到1个相关文档的查询比例")
        
        # 性能评估
        print("\n性能评估:")
        print("-" * 30)
        if 'Recall@1' in metrics:
            r1 = metrics['Recall@1']
            if r1 > 0.8:
                print("✅ Recall@1 > 80%: 优秀的检索性能")
            elif r1 > 0.6:
                print("⚡ Recall@1 > 60%: 良好的检索性能")
            elif r1 > 0.4:
                print("⚠️  Recall@1 > 40%: 需要优化检索策略")
            else:
                print("❌ Recall@1 < 40%: 检索性能较差，需要重新设计")
        
        if 'MRR' in metrics:
            mrr = metrics['MRR']
            if mrr > 0.7:
                print("✅ MRR > 0.7: 相关文档排名靠前")
            elif mrr > 0.5:
                print("⚡ MRR > 0.5: 排名性能良好")
            else:
                print("⚠️  MRR < 0.5: 相关文档排名偏后")


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    qa_file = project_root / "data" / "test" / "test_qa.jsonl"
    
    if not qa_file.exists():
        print(f"错误: 找不到QA数据集文件 {qa_file}")
        return
    
    # 创建评估器
    evaluator = SimpleRetrievalEvaluator()
    
    # 加载数据集
    print("加载QA数据集...")
    qa_dataset = evaluator.load_qa_dataset(str(qa_file))
    
    print(f"成功加载 {len(qa_dataset)} 个QA对")
    
    # 方案A：使用模拟检索器演示
    print("\n" + "="*60)
    print("方案A: 使用模拟检索器评估")
    print("="*60)
    
    mock_metrics = evaluator.evaluate_with_mock_retriever(qa_dataset)
    evaluator.print_detailed_results(mock_metrics, qa_dataset)
    
    # 如果检索器可用，也运行真实评估
    try:
        # 尝试加载真实检索器
        print("\n" + "="*60)
        print("尝试使用真实检索器评估...")
        print("="*60)
        
        config = Config()
        embedding_model = EmbeddingModel(config.embedding_model_name)
        retriever = HybridRetriever(
            index_path=str(project_root / "models" / "faiss_index.bin"),
            documents_path=str(project_root / "models" / "documents.pkl"),
            embedding_model=embedding_model
        )
        
        evaluator.retriever = retriever
        real_metrics = evaluator.evaluate_with_real_retriever(qa_dataset)
        
        print("\n" + "="*60)
        print("真实检索器评估结果")
        print("="*60)
        evaluator.print_detailed_results(real_metrics, qa_dataset)
        
        # 比较两种结果
        print("\n" + "="*60)
        print("模拟 vs 真实检索器对比")
        print("="*60)
        
        comparison_metrics = ['Recall@1', 'Recall@3', 'Recall@5', 'MRR']
        print(f"{'指标':<12} {'模拟检索器':<12} {'真实检索器':<12} {'差值':<10}")
        print("-" * 50)
        
        for metric in comparison_metrics:
            if metric in mock_metrics and metric in real_metrics:
                mock_val = mock_metrics[metric]
                real_val = real_metrics[metric]
                diff = real_val - mock_val
                print(f"{metric:<12} {mock_val:<12.4f} {real_val:<12.4f} {diff:>+8.4f}")
        
    except Exception as e:
        print(f"\n无法加载真实检索器: {e}")
        print("这是正常的，如果您还没有构建索引或安装依赖")
    
    # 保存结果
    results_file = project_root / "results" / "retrieval_evaluation_simple.json"
    results_file.parent.mkdir(exist_ok=True)
    
    evaluation_results = {
        'dataset_info': {
            'total_queries': len(qa_dataset),
            'categories': {}
        },
        'mock_retriever_metrics': mock_metrics
    }
    
    # 统计类别
    for qa in qa_dataset:
        cat = qa.get('category', 'unknown')
        evaluation_results['dataset_info']['categories'][cat] = \
            evaluation_results['dataset_info']['categories'].get(cat, 0) + 1
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()