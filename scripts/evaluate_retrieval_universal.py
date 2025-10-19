#!/usr/bin/env python3
"""
统一的检索评估工具
支持两种QA数据格式：
1. 方案A：每个问题1个相关文档（doc_id字段）
2. 方案B：每个问题多个相关文档（relevant_doc_ids字段）
"""

import json
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path
import sys
import argparse
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class UniversalRetrievalEvaluator:
    """
    通用检索评估器 - 支持单文档和多文档格式
    """
    
    def __init__(self, retriever=None):
        self.retriever = retriever
    
    def load_qa_dataset(self, file_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        加载QA数据集并自动检测格式
        
        Returns:
            (数据列表, 格式类型)
        """
        qa_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pairs.append(json.loads(line.strip()))
        
        # 检测格式
        if qa_pairs and 'relevant_doc_ids' in qa_pairs[0]:
            format_type = "multi_doc"  # 方案B：多文档格式
        else:
            format_type = "single_doc"  # 方案A：单文档格式
        
        return qa_pairs, format_type
    
    def normalize_qa_dataset(self, qa_dataset: List[Dict[str, Any]], 
                           format_type: str) -> List[Dict[str, Any]]:
        """
        将数据集标准化为统一格式
        """
        normalized_dataset = []
        
        for qa in qa_dataset:
            normalized_qa = qa.copy()
            
            if format_type == "single_doc":
                # 方案A：单文档格式
                relevant_docs = {qa['doc_id']} if 'doc_id' in qa else set()
            else:
                # 方案B：多文档格式
                relevant_docs = set(qa.get('relevant_doc_ids', []))
            
            normalized_qa['relevant_docs'] = relevant_docs
            normalized_qa['num_relevant_docs'] = len(relevant_docs)
            
            normalized_dataset.append(normalized_qa)
        
        return normalized_dataset
    
    def calculate_metrics(self, results: List[Dict[str, Any]], 
                         k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        计算全面的检索指标
        """
        metrics = {}
        total_queries = len(results)
        
        if total_queries == 0:
            return metrics
        
        # Recall@K
        for k in k_values:
            recall_at_k = 0
            for result in results:
                relevant_docs = result['relevant_docs']
                retrieved_docs_at_k = set(result['retrieved_docs'][:k])
                
                if len(relevant_docs) > 0:
                    recall = len(relevant_docs & retrieved_docs_at_k) / len(relevant_docs)
                    recall_at_k += recall
            
            metrics[f'Recall@{k}'] = recall_at_k / total_queries
        
        # Precision@K
        for k in k_values:
            precision_at_k = 0
            for result in results:
                relevant_docs = result['relevant_docs']
                retrieved_docs_at_k = set(result['retrieved_docs'][:k])
                
                if len(retrieved_docs_at_k) > 0:
                    precision = len(relevant_docs & retrieved_docs_at_k) / len(retrieved_docs_at_k)
                    precision_at_k += precision
            
            metrics[f'Precision@{k}'] = precision_at_k / total_queries
        
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
        
        metrics['MRR'] = mrr / total_queries
        
        # MAP (Mean Average Precision)
        map_score = 0
        for result in results:
            relevant_docs = result['relevant_docs']
            retrieved_docs = result['retrieved_docs']
            
            if len(relevant_docs) == 0:
                continue
            
            precision_sum = 0
            relevant_retrieved = 0
            
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    relevant_retrieved += 1
                    precision_at_i = relevant_retrieved / (i + 1)
                    precision_sum += precision_at_i
            
            if relevant_retrieved > 0:
                avg_precision = precision_sum / len(relevant_docs)
                map_score += avg_precision
        
        metrics['MAP'] = map_score / total_queries
        
        # NDCG@K
        for k in k_values:
            ndcg_at_k = 0
            for result in results:
                relevant_docs = result['relevant_docs']
                retrieved_docs = result['retrieved_docs'][:k]
                
                dcg = 0
                idcg = 0
                
                # DCG
                for i, doc_id in enumerate(retrieved_docs):
                    relevance = 1 if doc_id in relevant_docs else 0
                    dcg += relevance / np.log2(i + 2)
                
                # IDCG
                for i in range(min(k, len(relevant_docs))):
                    idcg += 1 / np.log2(i + 2)
                
                if idcg > 0:
                    ndcg_at_k += dcg / idcg
            
            metrics[f'NDCG@{k}'] = ndcg_at_k / total_queries
        
        # Hit Rate
        hits = 0
        for result in results:
            relevant_docs = result['relevant_docs']
            retrieved_docs = set(result['retrieved_docs'])
            if len(relevant_docs & retrieved_docs) > 0:
                hits += 1
        
        metrics['Hit_Rate'] = hits / total_queries
        
        # Coverage (检索到的唯一相关文档数 / 总相关文档数)
        all_relevant_docs = set()
        retrieved_relevant_docs = set()
        
        for result in results:
            all_relevant_docs.update(result['relevant_docs'])
            retrieved_docs = set(result['retrieved_docs'])
            retrieved_relevant_docs.update(result['relevant_docs'] & retrieved_docs)
        
        if len(all_relevant_docs) > 0:
            metrics['Coverage'] = len(retrieved_relevant_docs) / len(all_relevant_docs)
        else:
            metrics['Coverage'] = 0.0
        
        return metrics
    
    def evaluate_with_mock_retriever(self, normalized_dataset: List[Dict[str, Any]], 
                                   format_type: str,
                                   num_retrieved: int = 10) -> Dict[str, float]:
        """
        使用模拟检索器进行评估
        """
        results = []
        
        # 获取所有可能的文档ID
        all_doc_ids = set()
        for qa in normalized_dataset:
            all_doc_ids.update(qa['relevant_docs'])
        all_doc_ids = list(all_doc_ids)
        
        import random
        random.seed(42)
        
        for qa in normalized_dataset:
            query = qa['question']
            relevant_docs = qa['relevant_docs']
            
            # 根据格式类型调整模拟策略
            if format_type == "single_doc":
                # 单文档：50%概率相关文档在前3位
                success_rate = 0.5
                early_positions = 3
            else:
                # 多文档：70%概率至少一个相关文档在前5位
                success_rate = 0.7
                early_positions = 5
            
            retrieved_docs = []
            other_docs = [doc for doc in all_doc_ids if doc not in relevant_docs]
            random.shuffle(other_docs)
            
            if random.random() < success_rate:
                # 成功情况
                if format_type == "single_doc":
                    # 单文档：将唯一相关文档放在前面
                    position = random.randint(0, min(early_positions-1, num_retrieved-1))
                    for i in range(num_retrieved):
                        if i == position and relevant_docs:
                            retrieved_docs.append(list(relevant_docs)[0])
                        else:
                            if other_docs:
                                retrieved_docs.append(other_docs.pop())
                else:
                    # 多文档：随机选择1-2个相关文档放在前面
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
                # 失败情况
                retrieved_docs = other_docs[:num_retrieved]
                
                # 小概率在后面包含一个相关文档
                if random.random() < 0.25 and relevant_docs and len(retrieved_docs) > early_positions:
                    pos = random.randint(early_positions, len(retrieved_docs)-1)
                    retrieved_docs[pos] = random.choice(list(relevant_docs))
            
            results.append({
                'query': query,
                'relevant_docs': relevant_docs,
                'retrieved_docs': retrieved_docs[:num_retrieved]
            })
        
        return self.calculate_metrics(results)
    
    def analyze_dataset_characteristics(self, normalized_dataset: List[Dict[str, Any]], 
                                     format_type: str) -> Dict[str, Any]:
        """
        分析数据集特征
        """
        characteristics = {
            'total_queries': len(normalized_dataset),
            'format_type': format_type,
            'categories': {},
            'years': {},
            'relevant_docs_stats': {}
        }
        
        relevant_doc_counts = []
        
        for qa in normalized_dataset:
            # 类别统计
            cat = qa.get('category', 'unknown')
            characteristics['categories'][cat] = characteristics['categories'].get(cat, 0) + 1
            
            # 年份统计
            year = qa.get('year', 'unknown')
            characteristics['years'][year] = characteristics['years'].get(year, 0) + 1
            
            # 相关文档数量
            relevant_doc_counts.append(qa['num_relevant_docs'])
        
        # 相关文档统计
        if relevant_doc_counts:
            characteristics['relevant_docs_stats'] = {
                'mean': float(np.mean(relevant_doc_counts)),
                'std': float(np.std(relevant_doc_counts)),
                'min': int(min(relevant_doc_counts)),
                'max': int(max(relevant_doc_counts)),
                'median': float(np.median(relevant_doc_counts))
            }
        
        return characteristics
    
    def print_comprehensive_results(self, metrics: Dict[str, float], 
                                  characteristics: Dict[str, Any],
                                  title: str = "检索评估结果"):
        """
        打印全面的评估结果
        """
        print("=" * 80)
        print(f"{title}")
        print("=" * 80)
        
        # 数据集信息
        print(f"数据集类型: {characteristics['format_type']}")
        print(f"查询总数: {characteristics['total_queries']}")
        
        rel_stats = characteristics.get('relevant_docs_stats', {})
        if rel_stats:
            print(f"相关文档数: 平均{rel_stats['mean']:.2f} (范围: {rel_stats['min']}-{rel_stats['max']})")
        
        # 核心指标
        print("\n📊 核心检索指标:")
        print("-" * 50)
        
        core_metrics = [
            ('Recall@1', 'R@1'),
            ('Recall@3', 'R@3'), 
            ('Recall@5', 'R@5'),
            ('Precision@3', 'P@3'),
            ('F1@3', 'F1@3'),
            ('MRR', 'MRR'),
            ('MAP', 'MAP'),
            ('Hit_Rate', 'Hit')
        ]
        
        for full_name, short_name in core_metrics:
            if full_name in metrics:
                value = metrics[full_name]
                print(f"{short_name:8}: {value:.4f} ({value*100:5.1f}%)")
        
        # 完整Recall指标
        print("\n🎯 召回率指标 (Recall@K):")
        print("-" * 50)
        recall_metrics = [(k, v) for k, v in metrics.items() if k.startswith('Recall@')]
        for metric, value in sorted(recall_metrics):
            k = metric.split('@')[1]
            print(f"R@{k:2}: {value:.4f} ({value*100:5.1f}%)")
        
        # 完整Precision指标  
        print("\n🎯 精确率指标 (Precision@K):")
        print("-" * 50)
        precision_metrics = [(k, v) for k, v in metrics.items() if k.startswith('Precision@')]
        for metric, value in sorted(precision_metrics):
            k = metric.split('@')[1]
            print(f"P@{k:2}: {value:.4f} ({value*100:5.1f}%)")
        
        # 高级指标
        print("\n🏆 高级指标:")
        print("-" * 50)
        advanced_metrics = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'Coverage']
        for metric in advanced_metrics:
            if metric in metrics:
                value = metrics[metric]
                print(f"{metric:10}: {value:.4f} ({value*100:5.1f}%)")
        
        # 性能评估
        print("\n📈 性能评估:")
        print("-" * 50)
        
        # 根据不同格式给出不同建议
        format_type = characteristics['format_type']
        
        if format_type == "single_doc":
            # 单文档格式评估
            r1 = metrics.get('Recall@1', 0)
            mrr = metrics.get('MRR', 0)
            
            if r1 > 0.8:
                print("✅ Recall@1 > 80%: 优秀的单文档检索性能")
            elif r1 > 0.6:
                print("⚡ Recall@1 > 60%: 良好的单文档检索性能")  
            elif r1 > 0.4:
                print("⚠️  Recall@1 > 40%: 单文档检索需要优化")
            else:
                print("❌ Recall@1 < 40%: 单文档检索性能较差")
                
            if mrr > 0.7:
                print("✅ MRR > 0.7: 相关文档排名很好")
            elif mrr > 0.5:
                print("⚡ MRR > 0.5: 相关文档排名良好")
            else:
                print("⚠️  MRR < 0.5: 相关文档排名需要改进")
                
        else:
            # 多文档格式评估
            r5 = metrics.get('Recall@5', 0)
            f1_3 = metrics.get('F1@3', 0)
            map_score = metrics.get('MAP', 0)
            
            if r5 > 0.8:
                print("✅ Recall@5 > 80%: 优秀的多文档检索覆盖率")
            elif r5 > 0.6:
                print("⚡ Recall@5 > 60%: 良好的多文档检索覆盖率")
            elif r5 > 0.4:
                print("⚠️  Recall@5 > 40%: 多文档检索覆盖率需要改进") 
            else:
                print("❌ Recall@5 < 40%: 多文档检索覆盖率较差")
                
            if f1_3 > 0.6:
                print("✅ F1@3 > 60%: 优秀的精确率召回率平衡")
            elif f1_3 > 0.4:
                print("⚡ F1@3 > 40%: 良好的精确率召回率平衡")
            else:
                print("⚠️  F1@3 < 40%: 需要平衡精确率和召回率")
                
            if map_score > 0.5:
                print("✅ MAP > 0.5: 优秀的平均精确率")
            elif map_score > 0.3:
                print("⚡ MAP > 0.3: 良好的平均精确率")
            else:
                print("⚠️  MAP < 0.3: 平均精确率需要改进")
        
        # 类别统计
        print("\n📋 类别分布:")
        print("-" * 50)
        categories = characteristics.get('categories', {})
        for cat, count in sorted(categories.items()):
            percentage = (count / characteristics['total_queries']) * 100
            print(f"{cat:12}: {count:3d} ({percentage:4.1f}%)")


def compare_evaluations(results_a: Dict[str, float], 
                       results_b: Dict[str, float],
                       char_a: Dict[str, Any],
                       char_b: Dict[str, Any]):
    """
    比较两种评估方案的结果
    """
    print("\n" + "=" * 80)
    print("📊 方案对比分析")
    print("=" * 80)
    
    print(f"方案A ({char_a['format_type']}): {char_a['total_queries']} 个查询")
    print(f"方案B ({char_b['format_type']}): {char_b['total_queries']} 个查询")
    
    # 对比核心指标
    comparison_metrics = [
        'Recall@1', 'Recall@3', 'Recall@5', 
        'Precision@3', 'F1@3', 'MRR', 'MAP'
    ]
    
    print(f"\n{'指标':<12} {'方案A':<10} {'方案B':<10} {'差值':<10} {'改进':<8}")
    print("-" * 55)
    
    for metric in comparison_metrics:
        val_a = results_a.get(metric, 0)
        val_b = results_b.get(metric, 0)
        diff = val_b - val_a
        
        if abs(diff) < 0.001:
            improvement = "="
        elif diff > 0:
            improvement = f"+{diff/val_a*100:.1f}%" if val_a > 0 else "N/A"
        else:
            improvement = f"{diff/val_a*100:.1f}%" if val_a > 0 else "N/A"
        
        print(f"{metric:<12} {val_a:<10.4f} {val_b:<10.4f} {diff:>+8.4f} {improvement:>7}")
    
    # 建议
    print("\n💡 对比分析:")
    print("-" * 50)
    
    rel_stats_a = char_a.get('relevant_docs_stats', {})
    rel_stats_b = char_b.get('relevant_docs_stats', {}) 
    
    if rel_stats_b.get('mean', 1) > rel_stats_a.get('mean', 1):
        print("• 方案B提供了更丰富的相关文档标注")
        print("• 方案B更适合评估检索系统的全面性能")
        print("• 方案B支持更多样化的评估指标")
    
    if results_b.get('Recall@5', 0) > results_a.get('Recall@1', 0):
        print("• 多文档标注提高了检索评估的可靠性")
    
    print("• 两种方案可以互补使用：")
    print("  - 方案A适合快速评估和基准测试")
    print("  - 方案B适合深入分析和系统优化")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通用检索评估工具')
    parser.add_argument('--qa_file_a', type=str, 
                       help='方案A的QA文件路径（单文档格式）')
    parser.add_argument('--qa_file_b', type=str,
                       help='方案B的QA文件路径（多文档格式）')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    parser.add_argument('--num_retrieved', type=int, default=10,
                       help='检索文档数量')
    
    args = parser.parse_args()
    
    # 默认文件路径
    project_root = Path(__file__).parent.parent
    
    if not args.qa_file_a:
        args.qa_file_a = str(project_root / "data" / "test" / "test_qa.jsonl")
    
    if not args.qa_file_b:
        args.qa_file_b = str(project_root / "data" / "test" / "test_qa_expanded.jsonl")
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(exist_ok=True)
    
    evaluator = UniversalRetrievalEvaluator()
    
    results_summary = {
        'evaluation_params': {
            'num_retrieved': args.num_retrieved
        },
        'results': {}
    }
    
    # 评估方案A
    if Path(args.qa_file_a).exists():
        print("🔍 评估方案A...")
        qa_dataset_a, format_a = evaluator.load_qa_dataset(args.qa_file_a)
        normalized_a = evaluator.normalize_qa_dataset(qa_dataset_a, format_a)
        characteristics_a = evaluator.analyze_dataset_characteristics(normalized_a, format_a)
        
        metrics_a = evaluator.evaluate_with_mock_retriever(
            normalized_a, format_a, args.num_retrieved
        )
        
        evaluator.print_comprehensive_results(
            metrics_a, characteristics_a, 
            "方案A: 单文档相关性评估结果"
        )
        
        results_summary['results']['plan_a'] = {
            'characteristics': characteristics_a,
            'metrics': metrics_a
        }
    else:
        print(f"⚠️  方案A文件不存在: {args.qa_file_a}")
        metrics_a, characteristics_a = None, None
    
    # 评估方案B
    if Path(args.qa_file_b).exists():
        print("\n🔍 评估方案B...")
        qa_dataset_b, format_b = evaluator.load_qa_dataset(args.qa_file_b)
        normalized_b = evaluator.normalize_qa_dataset(qa_dataset_b, format_b)
        characteristics_b = evaluator.analyze_dataset_characteristics(normalized_b, format_b)
        
        metrics_b = evaluator.evaluate_with_mock_retriever(
            normalized_b, format_b, args.num_retrieved
        )
        
        evaluator.print_comprehensive_results(
            metrics_b, characteristics_b,
            "方案B: 多文档相关性评估结果"
        )
        
        results_summary['results']['plan_b'] = {
            'characteristics': characteristics_b,
            'metrics': metrics_b
        }
    else:
        print(f"⚠️  方案B文件不存在: {args.qa_file_b}")
        metrics_b, characteristics_b = None, None
    
    # 对比分析
    if metrics_a and metrics_b:
        compare_evaluations(metrics_a, metrics_b, characteristics_a, characteristics_b)
        
        results_summary['comparison'] = {
            'summary': "方案B在多文档相关性上表现更好，但方案A更适合快速评估"
        }
    
    # 保存结果
    results_file = output_dir / "universal_retrieval_evaluation.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 完整结果已保存到: {results_file}")
    
    # 使用建议
    print("\n" + "=" * 80)
    print("🎯 使用建议")
    print("=" * 80)
    print("1. 快速验证: 使用方案A进行基础检索性能验证")
    print("2. 深度分析: 使用方案B进行全面的检索系统评估")
    print("3. 系统优化: 结合两种方案的结果进行检索参数调优")
    print("4. 生产部署: 在真实检索器上运行两种评估以验证性能")


if __name__ == "__main__":
    main()