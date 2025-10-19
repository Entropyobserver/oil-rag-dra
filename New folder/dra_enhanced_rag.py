"""
DRA-Enhanced RAG System

This module integrates Dynamic Rank Adaptation (DRA) with the existing RAG pipeline.
It provides adaptive parameter selection based on query complexity analysis.

The DRA-RAG system dynamically adjusts retrieval and generation parameters to
optimize performance for different query types in the oil & gas domain.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Import existing RAG components
try:
    from src.retrieval.hybrid_retriever import HybridRetriever
except ImportError:
    # Fallback import path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from hybrid_retriever import HybridRetriever

try:
    from smart_answer_generator import SmartAnswerGenerator
except ImportError:
    logger.warning("SmartAnswerGenerator not found, using fallback")
    SmartAnswerGenerator = None

# Import DRA components
try:
    from src.models.dra_controller import DRAController, DRAParameters, ComplexityLevel
    from src.models.dynamic_lora import DynamicLoRAModel, LoRAConfig
except ImportError:
    # Local import fallback
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))
    try:
        from dra_controller import DRAController, DRAParameters, ComplexityLevel
        from dynamic_lora import DynamicLoRAModel, LoRAConfig
    except ImportError as e:
        logger.error(f"Could not import DRA components: {e}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DRAEnhancedRAG:
    """
    DRA-Enhanced RAG System
    
    Combines Dynamic Rank Adaptation with RAG pipeline for adaptive
    performance optimization based on query complexity.
    """
    
    def __init__(
        self,
        index_path: str,
        documents_path: str,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_reranking: bool = True
    ):
        """
        Initialize DRA-Enhanced RAG system.
        
        Args:
            index_path: Path to FAISS index
            documents_path: Path to documents pickle file
            embedder_model: Sentence transformer model name
            enable_reranking: Whether to enable reranking
        """
        
        # Initialize DRA Controller
        self.dra_controller = DRAController()
        
        # Initialize base RAG components
        self.retriever = HybridRetriever(
            index_path=index_path,
            documents_path=documents_path,
            embedder_model=embedder_model,
            enable_reranking=enable_reranking
        )
        
        self.answer_generator = SmartAnswerGenerator()
        
        # Initialize Dynamic LoRA model (if available)
        self.dynamic_lora = None
        self._initialize_lora_model()
        
        # Performance tracking
        self.query_history = []
        self.performance_stats = {
            'total_queries': 0,
            'by_complexity': {level.value: {'count': 0, 'avg_time': 0.0, 'avg_f1': 0.0} 
                            for level in ComplexityLevel},
            'parameter_adaptations': 0
        }
        
        logger.info("DRA-Enhanced RAG system initialized")
    
    def _initialize_lora_model(self):
        """Initialize Dynamic LoRA model if PyTorch is available."""
        try:
            self.dynamic_lora = DynamicLoRAModel(
                embedding_dim=384,  # Match sentence transformer dimension
                hidden_dim=256,
                num_layers=2
            )
            logger.info("Dynamic LoRA model initialized")
        except Exception as e:
            logger.warning(f"Dynamic LoRA not available: {e}")
            self.dynamic_lora = None
    
    def search_and_answer(
        self, 
        query: str, 
        return_sources: bool = True,
        override_params: Optional[DRAParameters] = None
    ) -> Dict[str, Any]:
        """
        Search and generate answer with DRA optimization.
        
        Args:
            query: User query
            return_sources: Whether to return source documents
            override_params: Optional parameter override
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        start_time = time.time()
        
        # Step 1: Analyze query and adapt parameters
        if override_params:
            dra_params = override_params
            complexity_level = override_params.complexity_level
            logger.info(f"Using override parameters: {complexity_level.value}")
        else:
            dra_params = self.dra_controller.adapt_parameters(query)
            complexity_level = dra_params.complexity_level
            self.performance_stats['parameter_adaptations'] += 1
        
        # Step 2: Configure Dynamic LoRA if available
        if self.dynamic_lora:
            self.dynamic_lora.adapt_from_parameters(dra_params)
        
        # Step 3: Adaptive document retrieval
        retrieval_results = self._adaptive_retrieval(query, dra_params)
        
        # Step 4: Adaptive answer generation
        generation_results = self._adaptive_generation(
            query, retrieval_results, dra_params
        )
        
        # Step 5: Calculate performance metrics
        total_time = time.time() - start_time
        
        # Prepare response
        response = {
            'query': query,
            'answer': generation_results['answer'],
            'complexity_level': complexity_level.value,
            'complexity_score': self.dra_controller.adaptation_history[-1]['complexity_score'] if self.dra_controller.adaptation_history else 0.0,
            'response_time': total_time,
            'parameters_used': {
                'retrieval_k': dra_params.retrieval_k,
                'rerank_top_k': dra_params.rerank_top_k,
                'similarity_threshold': dra_params.similarity_threshold,
                'max_context_length': dra_params.max_context_length,
                'lora_rank': dra_params.lora_rank if self.dynamic_lora else None
            }
        }
        
        if return_sources:
            response['sources'] = retrieval_results.get('documents', [])
            response['source_count'] = len(response['sources'])
            response['relevance_scores'] = retrieval_results.get('scores', [])
        
        # Update performance tracking
        self._update_performance_stats(complexity_level, total_time, response)
        
        # Store query history
        self.query_history.append({
            'timestamp': time.time(),
            'query': query,
            'complexity': complexity_level.value,
            'response_time': total_time,
            'parameters': dra_params
        })
        
        logger.info(f"Query processed: {complexity_level.value} level, {total_time:.3f}s")
        
        return response
    
    def _adaptive_retrieval(
        self, 
        query: str, 
        params: DRAParameters
    ) -> Dict[str, Any]:
        """
        Perform adaptive document retrieval based on DRA parameters.
        
        Args:
            query: User query
            params: DRA parameters
            
        Returns:
            Retrieval results with documents and scores
        """
        
        # Configure retrieval parameters
        retrieval_config = {
            'k': params.retrieval_k,
            'similarity_threshold': params.similarity_threshold
        }
        
        # Perform retrieval
        try:
            results = self.retriever.retrieve(
                query=query,
                k=retrieval_config['k'],
                return_scores=True
            )
            
            # Filter by similarity threshold
            if 'scores' in results:
                filtered_docs = []
                filtered_scores = []
                
                for doc, score in zip(results['documents'], results['scores']):
                    if score >= retrieval_config['similarity_threshold']:
                        filtered_docs.append(doc)
                        filtered_scores.append(score)
                
                results['documents'] = filtered_docs[:params.rerank_top_k]
                results['scores'] = filtered_scores[:params.rerank_top_k]
            
            logger.debug(f"Retrieved {len(results['documents'])} documents above threshold")
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            results = {'documents': [], 'scores': []}
        
        return results
    
    def _adaptive_generation(
        self, 
        query: str, 
        retrieval_results: Dict[str, Any], 
        params: DRAParameters
    ) -> Dict[str, Any]:
        """
        Perform adaptive answer generation based on DRA parameters.
        
        Args:
            query: User query
            retrieval_results: Retrieved documents and scores
            params: DRA parameters
            
        Returns:
            Generation results with answer and metadata
        """
        
        documents = retrieval_results.get('documents', [])
        
        if not documents:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'method': 'fallback',
                'context_length': 0
            }
        
        # Adaptive context preparation
        context_text = self._prepare_adaptive_context(
            documents, params.max_context_length
        )
        
        # Generate answer using Smart Answer Generator
        try:
            answer = self.answer_generator.generate_answer(
                query=query,
                context=context_text,
                max_length=params.max_tokens,
                temperature=params.temperature
            )
            
            generation_method = 'smart_generation'
            
        except Exception as e:
            logger.warning(f"Smart generation failed: {e}, using fallback")
            
            # Fallback to simple concatenation
            answer = self._fallback_generation(query, documents[:3])
            generation_method = 'fallback'
        
        return {
            'answer': answer,
            'method': generation_method,
            'context_length': len(context_text),
            'documents_used': len(documents)
        }
    
    def _prepare_adaptive_context(
        self, 
        documents: List[Dict[str, Any]], 
        max_length: int
    ) -> str:
        """
        Prepare context adaptively based on max length constraint.
        
        Args:
            documents: List of retrieved documents
            max_length: Maximum context length
            
        Returns:
            Prepared context string
        """
        
        context_parts = []
        current_length = 0
        
        for doc in documents:
            doc_text = doc.get('content', doc.get('text', ''))
            
            # Estimate token length (rough approximation: 4 chars per token)
            estimated_tokens = len(doc_text) // 4
            
            if current_length + estimated_tokens <= max_length:
                context_parts.append(doc_text)
                current_length += estimated_tokens
            else:
                # Add partial document if space allows
                remaining_chars = (max_length - current_length) * 4
                if remaining_chars > 100:  # Only add if substantial content fits
                    partial_text = doc_text[:remaining_chars]
                    # Try to cut at sentence boundary
                    last_period = partial_text.rfind('. ')
                    if last_period > len(partial_text) * 0.7:  # If we can keep 70%+ content
                        partial_text = partial_text[:last_period + 1]
                    context_parts.append(partial_text)
                break
        
        return "\n\n".join(context_parts)
    
    def _fallback_generation(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Fallback answer generation method.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Generated answer
        """
        
        if not documents:
            return "No relevant information found."
        
        # Simple extraction of relevant sentences
        relevant_sentences = []
        
        for doc in documents:
            content = doc.get('content', doc.get('text', ''))
            sentences = content.split('. ')
            
            # Find sentences containing query keywords
            query_words = query.lower().split()
            
            for sentence in sentences[:3]:  # Limit to first 3 sentences per doc
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words):
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Combine and summarize
            combined_text = '. '.join(relevant_sentences[:5])  # Max 5 sentences
            return f"Based on the available information: {combined_text}"
        else:
            # Return first few sentences from first document
            first_doc = documents[0]
            content = first_doc.get('content', first_doc.get('text', ''))
            sentences = content.split('. ')[:2]
            return '. '.join(sentences) + '.'
    
    def _update_performance_stats(
        self, 
        complexity: ComplexityLevel, 
        response_time: float, 
        response: Dict[str, Any]
    ):
        """Update performance statistics."""
        
        self.performance_stats['total_queries'] += 1
        
        complexity_stats = self.performance_stats['by_complexity'][complexity.value]
        complexity_stats['count'] += 1
        
        # Update running average for response time
        current_avg_time = complexity_stats['avg_time']
        count = complexity_stats['count']
        complexity_stats['avg_time'] = (
            (current_avg_time * (count - 1) + response_time) / count
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance statistics and analysis
        """
        
        # DRA Controller stats
        dra_stats = self.dra_controller.get_adaptation_stats()
        
        # Query distribution analysis
        if self.query_history:
            recent_queries = self.query_history[-50:]  # Last 50 queries
            
            complexity_distribution = {}
            avg_times_by_complexity = {}
            
            for level in ComplexityLevel:
                level_queries = [q for q in recent_queries if q['complexity'] == level.value]
                complexity_distribution[level.value] = len(level_queries)
                
                if level_queries:
                    avg_times_by_complexity[level.value] = np.mean([
                        q['response_time'] for q in level_queries
                    ])
                else:
                    avg_times_by_complexity[level.value] = 0.0
        else:
            complexity_distribution = {level.value: 0 for level in ComplexityLevel}
            avg_times_by_complexity = {level.value: 0.0 for level in ComplexityLevel}
        
        # System efficiency analysis
        total_adaptations = dra_stats.get('total_adaptations', 0)
        cache_hit_rate = dra_stats.get('cache_hit_rate', 0.0)
        
        report = {
            'overview': {
                'total_queries_processed': self.performance_stats['total_queries'],
                'total_parameter_adaptations': total_adaptations,
                'cache_hit_rate': cache_hit_rate,
                'dynamic_lora_enabled': self.dynamic_lora is not None
            },
            'complexity_analysis': {
                'distribution': complexity_distribution,
                'average_response_times': avg_times_by_complexity,
                'performance_by_complexity': self.performance_stats['by_complexity']
            },
            'dra_controller_stats': dra_stats,
            'recent_performance': {
                'last_10_queries': self.query_history[-10:] if self.query_history else []
            }
        }
        
        return report
    
    def benchmark_complexity_levels(
        self, 
        test_queries: List[str], 
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark performance across different complexity levels.
        
        Args:
            test_queries: List of test queries
            iterations: Number of iterations per query
            
        Returns:
            Benchmark results
        """
        
        benchmark_results = {
            'queries_tested': len(test_queries),
            'iterations_per_query': iterations,
            'results_by_complexity': {level.value: [] for level in ComplexityLevel},
            'comparative_analysis': {}
        }
        
        logger.info(f"Starting DRA benchmark with {len(test_queries)} queries")
        
        for query in test_queries:
            # Test with adaptive parameters (default)
            adaptive_times = []
            for _ in range(iterations):
                start_time = time.time()
                result = self.search_and_answer(query, return_sources=False)
                adaptive_times.append(time.time() - start_time)
            
            complexity = result['complexity_level']
            
            # Store results
            benchmark_results['results_by_complexity'][complexity].extend(adaptive_times)
        
        # Calculate comparative analysis
        for complexity in ComplexityLevel:
            times = benchmark_results['results_by_complexity'][complexity.value]
            if times:
                benchmark_results['comparative_analysis'][complexity.value] = {
                    'count': len(times),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
        
        logger.info("DRA benchmark completed")
        return benchmark_results


# Convenience function for easy system initialization
def create_dra_rag_system(
    index_path: str = "models/faiss_index.bin",
    documents_path: str = "models/documents.pkl"
) -> DRAEnhancedRAG:
    """
    Create and initialize DRA-Enhanced RAG system with default paths.
    
    Args:
        index_path: Path to FAISS index
        documents_path: Path to documents pickle file
        
    Returns:
        Initialized DRA-Enhanced RAG system
    """
    
    return DRAEnhancedRAG(
        index_path=index_path,
        documents_path=documents_path
    )


if __name__ == "__main__":
    # Example usage
    print("DRA-Enhanced RAG System Test")
    print("=" * 40)
    
    # Test queries with different complexity levels
    test_queries = [
        "What is oil production?",  # Simple
        "How did Equinor's production change in 2023?",  # Medium  
        "Analyze the relationship between environmental regulations and operational efficiency in offshore drilling projects over the past three years."  # Complex
    ]
    
    try:
        # Initialize system
        dra_rag = create_dra_rag_system()
        
        # Test each query
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}] Testing Query: {query}")
            print("-" * 60)
            
            result = dra_rag.search_and_answer(query, return_sources=False)
            
            print(f"Complexity Level: {result['complexity_level']}")
            print(f"Response Time: {result['response_time']:.3f}s")
            print(f"Answer: {result['answer'][:200]}...")
            
            if 'parameters_used' in result:
                params = result['parameters_used']
                print(f"Parameters - K: {params['retrieval_k']}, Threshold: {params['similarity_threshold']}")
        
        # Generate performance report
        print(f"\n{'='*60}")
        print("Performance Report:")
        report = dra_rag.get_performance_report()
        
        overview = report['overview']
        print(f"Total Queries: {overview['total_queries_processed']}")
        print(f"Parameter Adaptations: {overview['total_parameter_adaptations']}")
        print(f"Cache Hit Rate: {overview['cache_hit_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Error: {e}")
        print("Make sure FAISS index and documents are available.")