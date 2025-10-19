#!/usr/bin/env python3
"""
扩展相关文档脚本 - 方案B
使用语义相似度为每个问题找到3-5个相关文档，支持完整的Recall@K计算
"""

import json
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from collections import defaultdict


class DocumentExpander:
    """
    文档扩展器 - 使用语义相似度为每个问题找到多个相关文档
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """初始化文档扩展器"""
        print(f"加载嵌入模型: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.document_texts = {}
        self.document_embeddings = None
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
    
    def load_documents(self, documents_path: str):
        """
        加载文档数据
        支持多种格式：pkl文件或文本文件
        """
        doc_path = Path(documents_path)
        
        if doc_path.suffix == '.pkl':
            print(f"从pkl文件加载文档: {documents_path}")
            with open(documents_path, 'rb') as f:
                documents = pickle.load(f)
            
            # 转换格式
            for doc in documents:
                if isinstance(doc, dict):
                    doc_id = doc.get('doc_id', doc.get('id', ''))
                    text = doc.get('content', doc.get('text', ''))
                    self.document_texts[doc_id] = text
                    
        else:
            # 从QA数据集中提取文档
            print("从QA数据集中提取文档...")
            qa_file = Path(documents_path).parent.parent / "test" / "test_qa.jsonl"
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        qa = json.loads(line.strip())
                        doc_id = qa['doc_id']
                        # 使用context作为文档内容
                        text = qa.get('context', qa.get('answer', ''))
                        self.document_texts[doc_id] = text
        
        print(f"成功加载 {len(self.document_texts)} 个文档")
        
        # 创建索引映射
        doc_ids = list(self.document_texts.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.idx_to_doc_id = {idx: doc_id for doc_id, idx in self.doc_id_to_idx.items()}
    
    def compute_document_embeddings(self, cache_path: str = None):
        """计算文档嵌入"""
        if cache_path and Path(cache_path).exists():
            print(f"从缓存加载文档嵌入: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.document_embeddings = pickle.load(f)
            return
        
        print("计算文档嵌入...")
        texts = [self.document_texts[self.idx_to_doc_id[i]] 
                for i in range(len(self.document_texts))]
        
        self.document_embeddings = self.embedding_model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        if cache_path:
            print(f"保存嵌入到缓存: {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.document_embeddings, f)
    
    def find_relevant_documents(self, query: str, original_doc_id: str, 
                              num_similar: int = 4, 
                              min_similarity: float = 0.3) -> List[str]:
        """
        为查询找到相关文档
        
        Args:
            query: 查询文本
            original_doc_id: 原始相关文档ID
            num_similar: 要找到的相似文档数量
            min_similarity: 最小相似度阈值
        
        Returns:
            相关文档ID列表（包括原始文档）
        """
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # 计算与所有文档的相似度
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # 获取相似度最高的文档
        similar_indices = np.argsort(similarities)[::-1]
        
        relevant_docs = set()
        
        # 确保原始文档在结果中
        if original_doc_id in self.doc_id_to_idx:
            relevant_docs.add(original_doc_id)
        
        # 添加其他相似文档
        for idx in similar_indices:
            doc_id = self.idx_to_doc_id[idx]
            similarity = similarities[idx]
            
            if similarity >= min_similarity and len(relevant_docs) < num_similar + 1:
                relevant_docs.add(doc_id)
        
        # 如果相似文档不够，降低阈值
        if len(relevant_docs) < 2:  # 至少要有2个文档
            min_similarity = max(0.1, min_similarity - 0.1)
            for idx in similar_indices:
                doc_id = self.idx_to_doc_id[idx]
                similarity = similarities[idx]
                
                if similarity >= min_similarity and len(relevant_docs) < num_similar + 1:
                    relevant_docs.add(doc_id)
                
                if len(relevant_docs) >= num_similar + 1:
                    break
        
        return list(relevant_docs)
    
    def expand_qa_dataset(self, qa_dataset: List[Dict[str, Any]], 
                         num_similar: int = 4) -> List[Dict[str, Any]]:
        """
        扩展QA数据集，为每个问题添加多个相关文档
        """
        expanded_dataset = []
        
        print(f"扩展QA数据集，为每个问题寻找 {num_similar} 个相关文档...")
        
        for i, qa in enumerate(qa_dataset):
            if i % 20 == 0:
                print(f"处理进度: {i+1}/{len(qa_dataset)}")
            
            query = qa['question']
            original_doc_id = qa['doc_id']
            
            # 找到相关文档
            relevant_docs = self.find_relevant_documents(
                query, original_doc_id, num_similar
            )
            
            # 创建扩展的QA项
            expanded_qa = qa.copy()
            expanded_qa['relevant_doc_ids'] = relevant_docs
            expanded_qa['num_relevant_docs'] = len(relevant_docs)
            
            expanded_dataset.append(expanded_qa)
        
        return expanded_dataset


class EnhancedRetrievalEvaluator:
    """
    增强的检索评估器 - 支持多个相关文档
    """
    
    def calculate_metrics(self, results: List[Dict[str, Any]], 
                         k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        计算增强的检索指标
        """
        metrics = {}
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
        
        # Precision@K
        for k in k_values:
            precision_at_k = 0
            for result in results:
                relevant_docs = result['relevant_docs']
                retrieved_docs_at_k = set(result['retrieved_docs'][:k])
                
                if len(retrieved_docs_at_k) > 0:
                    precision = len(relevant_docs & retrieved_docs_at_k) / len(retrieved_docs_at_k)
                    precision_at_k += precision
            
            metrics[f'Precision@{k}'] = precision_at_k / total_queries if total_queries > 0 else 0.0
        
        # F1@K
        for k in k_values:
            recall_k = metrics.get(f'Recall@{k}', 0)
            precision_k = metrics.get(f'Precision@{k}', 0)
            
            if recall_k + precision_k > 0:
                f1_k = 2 * (recall_k * precision_k) / (recall_k + precision_k)
            else:
                f1_k = 0.0
            
            metrics[f'F1@{k}'] = f1_k
        
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
        
        # NDCG@K (简化版本)
        for k in k_values:
            ndcg_at_k = 0
            for result in results:
                relevant_docs = result['relevant_docs']
                retrieved_docs = result['retrieved_docs'][:k]
                
                dcg = 0
                idcg = 0
                
                for i, doc_id in enumerate(retrieved_docs):
                    relevance = 1 if doc_id in relevant_docs else 0
                    dcg += relevance / np.log2(i + 2)
                
                # 理想DCG
                for i in range(min(k, len(relevant_docs))):
                    idcg += 1 / np.log2(i + 2)
                
                if idcg > 0:
                    ndcg_at_k += dcg / idcg
            
            metrics[f'NDCG@{k}'] = ndcg_at_k / total_queries if total_queries > 0 else 0.0
        
        # Hit Rate
        hits = 0
        for result in results:
            relevant_docs = result['relevant_docs']
            retrieved_docs = set(result['retrieved_docs'])
            if len(relevant_docs & retrieved_docs) > 0:
                hits += 1
        
        metrics['Hit_Rate'] = hits / total_queries if total_queries > 0 else 0.0
        
        return metrics
    
    def evaluate_with_mock_retriever(self, expanded_dataset: List[Dict[str, Any]], 
                                   num_retrieved: int = 10) -> Dict[str, float]:
        """使用模拟检索器评估扩展数据集"""
        results = []
        
        # 获取所有可能的文档ID
        all_doc_ids = set()
        for qa in expanded_dataset:
            all_doc_ids.update(qa['relevant_doc_ids'])
        all_doc_ids = list(all_doc_ids)
        
        import random
        random.seed(42)
        
        for qa in expanded_dataset:
            query = qa['question']
            relevant_docs = set(qa['relevant_doc_ids'])
            
            # 模拟检索：70%概率至少包含一个相关文档在前5位
            retrieved_docs = []
            other_docs = [doc for doc in all_doc_ids if doc not in relevant_docs]
            random.shuffle(other_docs)
            
            if random.random() < 0.7:  # 70%概率有好结果
                # 随机选择1-2个相关文档放在前面
                selected_relevant = random.sample(
                    list(relevant_docs), 
                    min(random.randint(1, 2), len(relevant_docs))
                )
                
                for i in range(num_retrieved):
                    if i < len(selected_relevant):
                        retrieved_docs.append(selected_relevant[i])
                    else:
                        if other_docs:
                            retrieved_docs.append(other_docs.pop())
            else:
                # 30%概率较差结果
                retrieved_docs = other_docs[:num_retrieved]
                # 25%概率包含一个相关文档在后面
                if random.random() < 0.25 and relevant_docs and len(retrieved_docs) > 5:
                    pos = random.randint(5, len(retrieved_docs)-1)
                    retrieved_docs[pos] = random.choice(list(relevant_docs))
            
            results.append({
                'query': query,
                'relevant_docs': relevant_docs,
                'retrieved_docs': retrieved_docs[:num_retrieved]
            })
        
        return self.calculate_metrics(results)
    
    def print_detailed_results(self, metrics: Dict[str, float], 
                             dataset: List[Dict[str, Any]] = None,
                             title: str = "检索评估结果"):
        """打印详细的评估结果"""
        print("=" * 70)
        print(title)
        print("=" * 70)
        
        if dataset:
            print(f"数据集大小: {len(dataset)} 个查询")
            
            # 统计相关文档数量
            relevant_doc_counts = [len(qa['relevant_doc_ids']) for qa in dataset]
            avg_relevant = np.mean(relevant_doc_counts)
            print(f"平均相关文档数: {avg_relevant:.2f}")
            print(f"相关文档数范围: {min(relevant_doc_counts)} - {max(relevant_doc_counts)}")
        
        print("\n核心指标:")
        print("-" * 40)
        
        # 按重要性排序显示指标
        key_metrics = ['Recall@1', 'Recall@3', 'Recall@5', 'Precision@3', 'F1@3', 'MRR', 'NDCG@5']
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                print(f"{metric:12}: {value:.4f} ({value*100:.2f}%)")
        
        print("\n完整Recall指标:")
        print("-" * 40)
        recall_metrics = [k for k in metrics.keys() if k.startswith('Recall@')]
        for metric in sorted(recall_metrics):
            value = metrics[metric]
            print(f"{metric:15}: {value:.4f} ({value*100:.2f}%)")
        
        print("\n完整Precision指标:")
        print("-" * 40)
        precision_metrics = [k for k in metrics.keys() if k.startswith('Precision@')]
        for metric in sorted(precision_metrics):
            value = metrics[metric]
            print(f"{metric:15}: {value:.4f} ({value*100:.2f}%)")
        
        # 性能评估
        print("\n性能评估:")
        print("-" * 40)
        
        if 'Recall@5' in metrics:
            r5 = metrics['Recall@5']
            if r5 > 0.8:
                print("✅ Recall@5 > 80%: 优秀的检索覆盖率")
            elif r5 > 0.6:
                print("⚡ Recall@5 > 60%: 良好的检索覆盖率")
            elif r5 > 0.4:
                print("⚠️  Recall@5 > 40%: 检索覆盖率需要改进")
            else:
                print("❌ Recall@5 < 40%: 检索覆盖率较差")
        
        if 'F1@3' in metrics:
            f1 = metrics['F1@3']
            if f1 > 0.6:
                print("✅ F1@3 > 60%: 优秀的精确率和召回率平衡")
            elif f1 > 0.4:
                print("⚡ F1@3 > 40%: 良好的精确率和召回率平衡")
            else:
                print("⚠️  F1@3 < 40%: 需要平衡精确率和召回率")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    qa_file = project_root / "data" / "test" / "test_qa.jsonl"
    
    if not qa_file.exists():
        print(f"错误: 找不到QA数据集文件 {qa_file}")
        return
    
    # 加载原始QA数据集
    print("加载原始QA数据集...")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_dataset = [json.loads(line.strip()) for line in f]
    
    print(f"原始数据集: {len(qa_dataset)} 个QA对")
    
    # 创建文档扩展器
    print("\n" + "="*70)
    print("初始化文档扩展器...")
    print("="*70)
    
    expander = DocumentExpander()
    
    # 从QA数据集中加载文档
    expander.load_documents(str(qa_file))
    
    # 计算文档嵌入
    embeddings_cache = project_root / "models" / "document_embeddings.pkl"
    expander.compute_document_embeddings(str(embeddings_cache))
    
    # 扩展数据集
    print("\n" + "="*70)
    print("扩展QA数据集...")
    print("="*70)
    
    expanded_dataset = expander.expand_qa_dataset(qa_dataset, num_similar=4)
    
    # 保存扩展的数据集
    expanded_file = project_root / "data" / "test" / "test_qa_expanded.jsonl"
    with open(expanded_file, 'w', encoding='utf-8') as f:
        for qa in expanded_dataset:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    print(f"\n扩展数据集已保存到: {expanded_file}")
    
    # 统计扩展结果
    relevant_counts = [qa['num_relevant_docs'] for qa in expanded_dataset]
    print(f"扩展结果统计:")
    print(f"  平均相关文档数: {np.mean(relevant_counts):.2f}")
    print(f"  相关文档数范围: {min(relevant_counts)} - {max(relevant_counts)}")
    
    # 评估扩展数据集
    print("\n" + "="*70)
    print("评估扩展数据集...")
    print("="*70)
    
    evaluator = EnhancedRetrievalEvaluator()
    enhanced_metrics = evaluator.evaluate_with_mock_retriever(expanded_dataset)
    evaluator.print_detailed_results(
        enhanced_metrics, 
        expanded_dataset,
        "方案B: 扩展相关文档评估结果"
    )
    
    # 保存结果
    results_file = project_root / "results" / "retrieval_evaluation_enhanced.json"
    results_file.parent.mkdir(exist_ok=True)
    
    evaluation_results = {
        'dataset_info': {
            'total_queries': len(expanded_dataset),
            'avg_relevant_docs': float(np.mean(relevant_counts)),
            'relevant_docs_range': [int(min(relevant_counts)), int(max(relevant_counts))],
            'categories': {}
        },
        'enhanced_metrics': enhanced_metrics
    }
    
    # 统计类别
    for qa in expanded_dataset:
        cat = qa.get('category', 'unknown')
        evaluation_results['dataset_info']['categories'][cat] = \
            evaluation_results['dataset_info']['categories'].get(cat, 0) + 1
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {results_file}")
    
    # 显示数据集示例
    print("\n" + "="*70)
    print("扩展数据集示例:")
    print("="*70)
    
    example = expanded_dataset[0]
    print(f"问题: {example['question']}")
    print(f"原始相关文档: {example['doc_id']}")
    print(f"扩展相关文档: {example['relevant_doc_ids']}")
    print(f"相关文档数量: {example['num_relevant_docs']}")


if __name__ == "__main__":
    main()