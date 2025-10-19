"""
DRA Performance Evaluation

Evaluate DRA system performance against baseline static parameters
using the existing RAG system and QA dataset.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import pickle

# Add paths for imports
sys.path.append('/mnt/d/J/Desktop/language_technology/course/projects_AI/oil_rag_dra')
sys.path.append('/mnt/d/J/Desktop/language_technology/course/projects_AI/oil_rag_dra/src/models')

try:
    from dra_controller import DRAController, DRAParameters, ComplexityLevel
    from hybrid_retriever import HybridRetriever  
    from smart_answer_generator import SmartAnswerGenerator
    print("✅ All components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some components may not be available, continuing with available ones...")


def load_qa_dataset(filepath: str = "evaluation_results/qa_dataset.json") -> List[Dict[str, Any]]:
    """Load QA dataset for evaluation."""
    
    if not os.path.exists(filepath):
        print(f"❌ QA dataset not found at {filepath}")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        qa_pairs = dataset.get('qa_pairs', [])
        print(f"✅ Loaded {len(qa_pairs)} QA pairs from dataset")
        return qa_pairs
        
    except Exception as e:
        print(f"❌ Error loading QA dataset: {e}")
        return []


def initialize_rag_system():
    """Initialize RAG system components."""
    
    try:
        # Initialize retriever
        retriever = HybridRetriever(
            index_path="models/faiss_index.bin",
            documents_path="models/documents.pkl"
        )
        
        # Initialize answer generator
        answer_generator = SmartAnswerGenerator()
        
        print("✅ RAG system initialized successfully")
        return retriever, answer_generator
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return None, None


def evaluate_with_static_parameters(
    qa_pairs: List[Dict[str, Any]],
    retriever: Any,
    answer_generator: Any,
    static_params: DRAParameters,
    max_queries: int = 50
) -> Dict[str, Any]:
    """Evaluate RAG system with static parameters."""
    
    print(f"\n🔧 Evaluating with static parameters ({static_params.complexity_level.value})...")
    
    results = []
    total_time = 0
    
    for i, qa_pair in enumerate(qa_pairs[:max_queries]):
        if i % 10 == 0:
            print(f"   Processing query {i+1}/{min(max_queries, len(qa_pairs))}")
        
        query = qa_pair['question']
        expected_answer = qa_pair['answer']
        
        start_time = time.time()
        
        try:
            # Retrieve documents with static parameters
            retrieval_results = retriever.retrieve(
                query=query,
                k=static_params.retrieval_k,
                return_scores=True
            )
            
            # Filter by similarity threshold
            filtered_docs = []
            if 'documents' in retrieval_results and 'scores' in retrieval_results:
                for doc, score in zip(retrieval_results['documents'], retrieval_results['scores']):
                    if score >= static_params.similarity_threshold:
                        filtered_docs.append(doc)
            
            # Generate answer
            if filtered_docs:
                context_text = "\n\n".join([
                    doc.get('content', doc.get('text', ''))[:static_params.max_context_length//len(filtered_docs)]
                    for doc in filtered_docs[:static_params.rerank_top_k]
                ])
                
                generated_answer = answer_generator.generate_answer(
                    query=query,
                    context=context_text,
                    max_length=static_params.max_tokens,
                    temperature=static_params.temperature
                )
            else:
                generated_answer = "No relevant information found."
            
            response_time = time.time() - start_time
            total_time += response_time
            
            # Calculate F1 score (simple token-based)
            f1_score = calculate_f1_score(generated_answer, expected_answer)
            
            results.append({
                'query': query,
                'generated_answer': generated_answer,
                'expected_answer': expected_answer,
                'f1_score': f1_score,
                'response_time': response_time,
                'documents_retrieved': len(filtered_docs),
                'parameters_used': {
                    'retrieval_k': static_params.retrieval_k,
                    'similarity_threshold': static_params.similarity_threshold,
                    'max_context_length': static_params.max_context_length
                }
            })
            
        except Exception as e:
            print(f"      Error processing query {i+1}: {e}")
            results.append({
                'query': query,
                'generated_answer': "Error generating answer",
                'expected_answer': expected_answer,
                'f1_score': 0.0,
                'response_time': 0.0,
                'error': str(e)
            })
    
    # Calculate aggregated metrics
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        avg_f1 = np.mean([r['f1_score'] for r in valid_results])
        avg_time = np.mean([r['response_time'] for r in valid_results])
        throughput = len(valid_results) / total_time if total_time > 0 else 0
    else:
        avg_f1 = 0.0
        avg_time = 0.0
        throughput = 0.0
    
    evaluation_results = {
        'parameter_type': 'static',
        'complexity_level': static_params.complexity_level.value,
        'total_queries': len(results),
        'successful_queries': len(valid_results),
        'avg_f1_score': avg_f1,
        'avg_response_time': avg_time,
        'throughput': throughput,
        'individual_results': results,
        'parameters': {
            'retrieval_k': static_params.retrieval_k,
            'similarity_threshold': static_params.similarity_threshold,
            'max_context_length': static_params.max_context_length,
            'temperature': static_params.temperature
        }
    }
    
    print(f"   ✅ Static evaluation complete: F1={avg_f1:.3f}, Time={avg_time:.3f}s")
    
    return evaluation_results


def evaluate_with_dra_parameters(
    qa_pairs: List[Dict[str, Any]],
    retriever: Any,
    answer_generator: Any,
    dra_controller: DRAController,
    max_queries: int = 50
) -> Dict[str, Any]:
    """Evaluate RAG system with DRA adaptive parameters."""
    
    print(f"\n🎯 Evaluating with DRA adaptive parameters...")
    
    results = []
    total_time = 0
    adaptation_times = []
    complexity_distribution = {level.value: 0 for level in ComplexityLevel}
    
    for i, qa_pair in enumerate(qa_pairs[:max_queries]):
        if i % 10 == 0:
            print(f"   Processing query {i+1}/{min(max_queries, len(qa_pairs))}")
        
        query = qa_pair['question']
        expected_answer = qa_pair['answer']
        
        start_time = time.time()
        
        try:
            # Adapt parameters using DRA
            adaptation_start = time.time()
            dra_params = dra_controller.adapt_parameters(query)
            adaptation_time = time.time() - adaptation_start
            adaptation_times.append(adaptation_time)
            
            complexity_distribution[dra_params.complexity_level.value] += 1
            
            # Retrieve documents with adaptive parameters
            retrieval_results = retriever.retrieve(
                query=query,
                k=dra_params.retrieval_k,
                return_scores=True
            )
            
            # Filter by similarity threshold
            filtered_docs = []
            if 'documents' in retrieval_results and 'scores' in retrieval_results:
                for doc, score in zip(retrieval_results['documents'], retrieval_results['scores']):
                    if score >= dra_params.similarity_threshold:
                        filtered_docs.append(doc)
            
            # Generate answer
            if filtered_docs:
                context_text = "\n\n".join([
                    doc.get('content', doc.get('text', ''))[:dra_params.max_context_length//len(filtered_docs)]
                    for doc in filtered_docs[:dra_params.rerank_top_k]
                ])
                
                generated_answer = answer_generator.generate_answer(
                    query=query,
                    context=context_text,
                    max_length=dra_params.max_tokens,
                    temperature=dra_params.temperature
                )
            else:
                generated_answer = "No relevant information found."
            
            response_time = time.time() - start_time
            total_time += response_time
            
            # Calculate F1 score
            f1_score = calculate_f1_score(generated_answer, expected_answer)
            
            # Get complexity score
            complexity_score = dra_controller.adaptation_history[-1]['complexity_score'] if dra_controller.adaptation_history else 0.0
            
            results.append({
                'query': query,
                'generated_answer': generated_answer,
                'expected_answer': expected_answer,
                'f1_score': f1_score,
                'response_time': response_time,
                'adaptation_time': adaptation_time,
                'complexity_level': dra_params.complexity_level.value,
                'complexity_score': complexity_score,
                'documents_retrieved': len(filtered_docs),
                'parameters_used': {
                    'retrieval_k': dra_params.retrieval_k,
                    'similarity_threshold': dra_params.similarity_threshold,
                    'max_context_length': dra_params.max_context_length,
                    'temperature': dra_params.temperature,
                    'lora_rank': dra_params.lora_rank
                }
            })
            
        except Exception as e:
            print(f"      Error processing query {i+1}: {e}")
            results.append({
                'query': query,
                'generated_answer': "Error generating answer",
                'expected_answer': expected_answer,
                'f1_score': 0.0,
                'response_time': 0.0,
                'error': str(e)
            })
    
    # Calculate aggregated metrics
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        avg_f1 = np.mean([r['f1_score'] for r in valid_results])
        avg_time = np.mean([r['response_time'] for r in valid_results])
        avg_adaptation_time = np.mean(adaptation_times) if adaptation_times else 0.0
        throughput = len(valid_results) / total_time if total_time > 0 else 0
    else:
        avg_f1 = 0.0
        avg_time = 0.0
        avg_adaptation_time = 0.0
        throughput = 0.0
    
    evaluation_results = {
        'parameter_type': 'dra',
        'total_queries': len(results),
        'successful_queries': len(valid_results),
        'avg_f1_score': avg_f1,
        'avg_response_time': avg_time,
        'avg_adaptation_time': avg_adaptation_time,
        'throughput': throughput,
        'complexity_distribution': complexity_distribution,
        'individual_results': results,
        'dra_controller_stats': dra_controller.get_adaptation_stats()
    }
    
    print(f"   ✅ DRA evaluation complete: F1={avg_f1:.3f}, Time={avg_time:.3f}s")
    
    return evaluation_results


def calculate_f1_score(generated: str, expected: str) -> float:
    """Calculate F1 score between generated and expected answers."""
    
    # Simple token-based F1 calculation
    generated_tokens = set(generated.lower().split())
    expected_tokens = set(expected.lower().split())
    
    if not expected_tokens:
        return 0.0
    
    # Calculate precision and recall
    intersection = generated_tokens & expected_tokens
    
    if not intersection:
        return 0.0
    
    precision = len(intersection) / len(generated_tokens) if generated_tokens else 0.0
    recall = len(intersection) / len(expected_tokens) if expected_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compare_performance(
    static_results: List[Dict[str, Any]], 
    dra_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare performance between static and DRA approaches."""
    
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    comparison = {
        'static_approaches': {},
        'dra_approach': dra_results,
        'comparative_analysis': {}
    }
    
    # Organize static results
    for result in static_results:
        complexity = result['complexity_level']
        comparison['static_approaches'][complexity] = result
    
    # Compare metrics
    dra_f1 = dra_results['avg_f1_score']
    dra_time = dra_results['avg_response_time']
    dra_throughput = dra_results['throughput']
    
    print(f"DRA Results:")
    print(f"  Average F1 Score: {dra_f1:.3f}")
    print(f"  Average Response Time: {dra_time:.3f}s")
    print(f"  Throughput: {dra_throughput:.1f} q/s")
    print(f"  Complexity Distribution: {dra_results['complexity_distribution']}")
    
    print(f"\nStatic Approaches:")
    
    best_static_f1 = 0
    best_static_approach = None
    
    for complexity, result in comparison['static_approaches'].items():
        static_f1 = result['avg_f1_score']
        static_time = result['avg_response_time']
        static_throughput = result['throughput']
        
        print(f"  {complexity.upper()}:")
        print(f"    F1 Score: {static_f1:.3f}")
        print(f"    Response Time: {static_time:.3f}s")
        print(f"    Throughput: {static_throughput:.1f} q/s")
        
        if static_f1 > best_static_f1:
            best_static_f1 = static_f1
            best_static_approach = complexity
    
    # Calculate improvements
    if best_static_f1 > 0:
        f1_improvement = ((dra_f1 - best_static_f1) / best_static_f1) * 100
    else:
        f1_improvement = 0
    
    comparison['comparative_analysis'] = {
        'best_static_approach': best_static_approach,
        'best_static_f1': best_static_f1,
        'dra_f1': dra_f1,
        'f1_improvement_percent': f1_improvement,
        'dra_vs_best_static': {
            'f1_delta': dra_f1 - best_static_f1,
            'time_delta': dra_time - comparison['static_approaches'][best_static_approach]['avg_response_time'] if best_static_approach else 0
        }
    }
    
    print(f"\n🏆 BEST PERFORMANCE COMPARISON:")
    print(f"  Best Static ({best_static_approach}): F1 = {best_static_f1:.3f}")
    print(f"  DRA Adaptive: F1 = {dra_f1:.3f}")
    print(f"  Improvement: {f1_improvement:+.1f}%")
    
    return comparison


def main():
    """Run DRA performance evaluation."""
    
    print("🚀 DRA PERFORMANCE EVALUATION")
    print("=" * 80)
    
    # Load QA dataset
    qa_pairs = load_qa_dataset()
    if not qa_pairs:
        print("❌ No QA dataset available for evaluation")
        return
    
    # Initialize RAG system
    retriever, answer_generator = initialize_rag_system()
    if not retriever or not answer_generator:
        print("❌ Could not initialize RAG system")
        return
    
    # Initialize DRA Controller
    dra_controller = DRAController()
    
    # Define static parameter configurations
    static_configurations = [
        DRAParameters.get_default_params(ComplexityLevel.SIMPLE),
        DRAParameters.get_default_params(ComplexityLevel.MEDIUM), 
        DRAParameters.get_default_params(ComplexityLevel.COMPLEX)
    ]
    
    # Evaluate static approaches
    static_results = []
    for static_params in static_configurations:
        result = evaluate_with_static_parameters(
            qa_pairs, retriever, answer_generator, static_params, max_queries=30
        )
        static_results.append(result)
    
    # Evaluate DRA approach
    dra_results = evaluate_with_dra_parameters(
        qa_pairs, retriever, answer_generator, dra_controller, max_queries=30
    )
    
    # Compare performance
    comparison = compare_performance(static_results, dra_results)
    
    # Save results
    os.makedirs('evaluation_results', exist_ok=True)
    
    all_results = {
        'timestamp': time.time(),
        'evaluation_summary': comparison,
        'static_results': static_results,
        'dra_results': dra_results,
        'dataset_info': {
            'total_qa_pairs': len(qa_pairs),
            'queries_evaluated': 30
        }
    }
    
    with open('evaluation_results/dra_performance_evaluation.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ EVALUATION COMPLETED")
    print(f"📊 Results saved to: evaluation_results/dra_performance_evaluation.json")
    
    # Final summary
    improvement = comparison['comparative_analysis']['f1_improvement_percent']
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   DRA shows {improvement:+.1f}% F1 improvement over best static approach")
    
    return comparison


if __name__ == "__main__":
    main()