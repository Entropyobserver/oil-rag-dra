"""
DRA System Test and Evaluation

This script tests the Dynamic Rank Adaptation system independently
and evaluates its performance compared to static parameter approaches.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models'))

try:
    from dra_controller import DRAController, DRAParameters, ComplexityLevel
    from dynamic_lora import DynamicLoRAModel
    print("✅ DRA components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_dra_controller():
    """Test DRA Controller functionality."""
    
    print("\n" + "="*60)
    print("TESTING DRA CONTROLLER")
    print("="*60)
    
    controller = DRAController()
    
    # Test queries with different complexity levels
    test_queries = [
        "What is oil?",
        "How much oil was produced?",
        "What are the main production methods?",
        "How did Equinor's production efficiency change over the past three years?",
        "Compare the environmental impact of offshore drilling versus onshore extraction methods.",
        "Analyze the correlation between oil prices, production costs, and regulatory compliance in the Norwegian continental shelf during 2020-2024."
    ]
    
    results = []
    
    print(f"Testing {len(test_queries)} queries...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}] Query: {query}")
        
        # Analyze complexity
        start_time = time.time()
        params = controller.adapt_parameters(query)
        analysis_time = time.time() - start_time
        
        # Get complexity score
        if controller.adaptation_history:
            complexity_score = controller.adaptation_history[-1]['complexity_score']
        else:
            complexity_score = 0.0
        
        result = {
            'query': query,
            'complexity_level': params.complexity_level.value,
            'complexity_score': complexity_score,
            'analysis_time': analysis_time,
            'parameters': {
                'retrieval_k': params.retrieval_k,
                'rerank_top_k': params.rerank_top_k,
                'similarity_threshold': params.similarity_threshold,
                'max_context_length': params.max_context_length,
                'temperature': params.temperature,
                'lora_rank': params.lora_rank
            }
        }
        
        results.append(result)
        
        print(f"   Complexity: {params.complexity_level.value} (score: {complexity_score:.3f})")
        print(f"   Parameters: K={params.retrieval_k}, threshold={params.similarity_threshold:.2f}")
        print(f"   Analysis time: {analysis_time:.4f}s")
        print()
    
    # Summary statistics
    print("ANALYSIS SUMMARY:")
    print("-" * 40)
    
    complexity_counts = {}
    avg_times = {}
    
    for level in ComplexityLevel:
        level_results = [r for r in results if r['complexity_level'] == level.value]
        complexity_counts[level.value] = len(level_results)
        
        if level_results:
            avg_times[level.value] = np.mean([r['analysis_time'] for r in level_results])
        else:
            avg_times[level.value] = 0.0
    
    print(f"Complexity Distribution:")
    for level, count in complexity_counts.items():
        print(f"  {level.upper()}: {count} queries")
    
    print(f"\nAverage Analysis Times:")
    for level, avg_time in avg_times.items():
        print(f"  {level.upper()}: {avg_time:.4f}s")
    
    # Get controller statistics
    stats = controller.get_adaptation_stats()
    print(f"\nController Statistics:")
    print(f"  Total adaptations: {stats.get('total_adaptations', 0)}")
    print(f"  Average complexity: {stats.get('average_complexity', 0):.3f}")
    print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
    
    return results


def test_dynamic_lora():
    """Test Dynamic LoRA model."""
    
    print("\n" + "="*60)
    print("TESTING DYNAMIC LORA MODEL")
    print("="*60)
    
    try:
        import torch
        
        # Create model
        model = DynamicLoRAModel(
            embedding_dim=384,
            hidden_dim=256,
            num_layers=2
        )
        
        print("✅ Dynamic LoRA model created successfully")
        
        # Test different complexity levels
        test_input = torch.randn(2, 384)  # Batch of 2 embeddings
        
        print(f"Input shape: {test_input.shape}")
        print("-" * 40)
        
        results = {}
        
        for complexity in ComplexityLevel:
            print(f"Testing {complexity.value.upper()} complexity...")
            
            # Set complexity level
            model.set_complexity(complexity)
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            inference_time = time.time() - start_time
            
            # Get model stats
            stats = model.get_model_stats()
            
            results[complexity.value] = {
                'output_shape': list(output.shape),
                'inference_time': inference_time,
                'active_lora_rank': stats['active_lora_rank'],
                'lora_parameters': stats['lora_parameters_by_complexity'][complexity.value]
            }
            
            print(f"  Output shape: {output.shape}")
            print(f"  LoRA rank: {stats['active_lora_rank']}")
            print(f"  LoRA parameters: {stats['lora_parameters_by_complexity'][complexity.value]:,}")
            print(f"  Inference time: {inference_time:.4f}s")
            print()
        
        # Model statistics
        final_stats = model.get_model_stats()
        print("MODEL STATISTICS:")
        print("-" * 30)
        print(f"Total parameters: {final_stats['total_parameters']:,}")
        print(f"Trainable parameters: {final_stats['trainable_parameters']:,}")
        print(f"Embedding dimension: {final_stats['embedding_dim']}")
        print(f"Hidden dimension: {final_stats['hidden_dim']}")
        print(f"Number of layers: {final_stats['num_layers']}")
        
        return results
        
    except ImportError:
        print("❌ PyTorch not available, skipping Dynamic LoRA test")
        return {}


def benchmark_parameter_adaptation():
    """Benchmark parameter adaptation performance."""
    
    print("\n" + "="*60)
    print("BENCHMARKING PARAMETER ADAPTATION")
    print("="*60)
    
    controller = DRAController()
    
    # Generate test queries
    base_queries = [
        "oil production",
        "environmental regulations drilling",
        "compare offshore onshore extraction methods efficiency costs environmental impact",
        "analyze correlation oil prices production costs regulatory compliance Norwegian continental shelf 2020 2024 considering market volatility geopolitical factors sustainability initiatives"
    ]
    
    # Create variations
    test_queries = []
    for base_query in base_queries:
        test_queries.append(base_query)
        test_queries.append(f"What is {base_query}?")
        test_queries.append(f"How does {base_query} work?")
        test_queries.append(f"Analyze the impact of {base_query} on operations.")
    
    print(f"Benchmarking with {len(test_queries)} queries...")
    print("-" * 40)
    
    # Benchmark adaptation performance
    adaptation_times = []
    complexity_predictions = []
    
    for query in test_queries:
        start_time = time.time()
        params = controller.adapt_parameters(query)
        adaptation_time = time.time() - start_time
        
        adaptation_times.append(adaptation_time)
        complexity_predictions.append(params.complexity_level.value)
    
    # Performance analysis
    print("BENCHMARK RESULTS:")
    print("-" * 30)
    print(f"Total queries processed: {len(test_queries)}")
    print(f"Average adaptation time: {np.mean(adaptation_times):.4f}s")
    print(f"Min adaptation time: {np.min(adaptation_times):.4f}s")
    print(f"Max adaptation time: {np.max(adaptation_times):.4f}s")
    print(f"Standard deviation: {np.std(adaptation_times):.4f}s")
    
    # Complexity distribution
    complexity_dist = {}
    for complexity in ComplexityLevel:
        count = complexity_predictions.count(complexity.value)
        complexity_dist[complexity.value] = {
            'count': count,
            'percentage': (count / len(complexity_predictions)) * 100
        }
    
    print(f"\nComplexity Distribution:")
    for level, stats in complexity_dist.items():
        print(f"  {level.upper()}: {stats['count']} queries ({stats['percentage']:.1f}%)")
    
    # Throughput calculation
    total_time = sum(adaptation_times)
    throughput = len(test_queries) / total_time
    
    print(f"\nThroughput: {throughput:.1f} queries/second")
    
    return {
        'total_queries': len(test_queries),
        'adaptation_times': adaptation_times,
        'complexity_distribution': complexity_dist,
        'throughput': throughput
    }


def compare_static_vs_dynamic_parameters():
    """Compare static vs dynamic parameter selection."""
    
    print("\n" + "="*60)
    print("STATIC vs DYNAMIC PARAMETER COMPARISON")
    print("="*60)
    
    controller = DRAController()
    
    # Test queries with known complexity levels
    test_cases = [
        ("What is oil?", "simple"),
        ("How much oil was produced in 2023?", "medium"),
        ("Analyze the long-term environmental and economic impacts of deep-water drilling technologies on Norwegian continental shelf operations.", "complex")
    ]
    
    # Static parameter sets (one-size-fits-all approaches)
    static_configs = {
        'conservative': DRAParameters.get_default_params(ComplexityLevel.SIMPLE),
        'balanced': DRAParameters.get_default_params(ComplexityLevel.MEDIUM),
        'aggressive': DRAParameters.get_default_params(ComplexityLevel.COMPLEX)
    }
    
    print("Comparing parameter selection strategies...")
    print("-" * 50)
    
    comparison_results = []
    
    for query, expected_complexity in test_cases:
        print(f"\nQuery: {query}")
        print(f"Expected complexity: {expected_complexity}")
        print("-" * 30)
        
        # Dynamic adaptation
        dynamic_params = controller.adapt_parameters(query)
        predicted_complexity = dynamic_params.complexity_level.value
        
        print(f"Dynamic prediction: {predicted_complexity}")
        
        result = {
            'query': query,
            'expected_complexity': expected_complexity,
            'predicted_complexity': predicted_complexity,
            'correct_prediction': predicted_complexity == expected_complexity,
            'parameter_comparison': {
                'dynamic': {
                    'retrieval_k': dynamic_params.retrieval_k,
                    'similarity_threshold': dynamic_params.similarity_threshold,
                    'max_context_length': dynamic_params.max_context_length,
                    'lora_rank': dynamic_params.lora_rank
                }
            }
        }
        
        # Compare with static approaches
        for config_name, static_params in static_configs.items():
            result['parameter_comparison'][f'static_{config_name}'] = {
                'retrieval_k': static_params.retrieval_k,
                'similarity_threshold': static_params.similarity_threshold,
                'max_context_length': static_params.max_context_length,
                'lora_rank': static_params.lora_rank
            }
        
        comparison_results.append(result)
        
        # Display comparison
        print("Parameter Comparison:")
        print(f"  Dynamic: K={dynamic_params.retrieval_k}, threshold={dynamic_params.similarity_threshold:.2f}, LoRA={dynamic_params.lora_rank}")
        
        for config_name, static_params in static_configs.items():
            print(f"  Static ({config_name}): K={static_params.retrieval_k}, threshold={static_params.similarity_threshold:.2f}, LoRA={static_params.lora_rank}")
    
    # Accuracy analysis
    correct_predictions = sum(1 for r in comparison_results if r['correct_prediction'])
    accuracy = correct_predictions / len(comparison_results)
    
    print(f"\n" + "="*50)
    print("COMPARISON SUMMARY:")
    print("="*50)
    print(f"Complexity prediction accuracy: {accuracy:.1%} ({correct_predictions}/{len(comparison_results)})")
    
    return comparison_results


def main():
    """Run comprehensive DRA system tests."""
    
    print("🚀 DYNAMIC RANK ADAPTATION (DRA) SYSTEM TEST")
    print("="*80)
    
    # Test 1: DRA Controller
    controller_results = test_dra_controller()
    
    # Test 2: Dynamic LoRA Model
    lora_results = test_dynamic_lora()
    
    # Test 3: Parameter Adaptation Benchmark
    benchmark_results = benchmark_parameter_adaptation()
    
    # Test 4: Static vs Dynamic Comparison
    comparison_results = compare_static_vs_dynamic_parameters()
    
    # Save results
    all_results = {
        'timestamp': time.time(),
        'controller_test': controller_results,
        'lora_test': lora_results,
        'benchmark': benchmark_results,
        'comparison': comparison_results
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    with open('evaluation_results/dra_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n" + "="*80)
    print("🎉 DRA SYSTEM TESTING COMPLETED")
    print("="*80)
    print(f"✅ All tests completed successfully")
    print(f"📊 Results saved to: evaluation_results/dra_test_results.json")
    
    # Summary
    if controller_results:
        complexity_dist = {}
        for level in ComplexityLevel:
            count = len([r for r in controller_results if r['complexity_level'] == level.value])
            complexity_dist[level.value] = count
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Queries processed: {len(controller_results)}")
        print(f"   Complexity distribution: {complexity_dist}")
        if benchmark_results:
            print(f"   Adaptation throughput: {benchmark_results['throughput']:.1f} q/s")
        if comparison_results:
            accuracy = sum(1 for r in comparison_results if r['correct_prediction']) / len(comparison_results)
            print(f"   Complexity prediction accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()