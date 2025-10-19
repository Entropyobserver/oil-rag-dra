"""
Dynamic Rank Adaptation Controller

This module implements the DRA (Dynamic Rank Adaptation) system that dynamically
adjusts model parameters based on query complexity analysis.

The DRA Controller analyzes incoming queries and selects appropriate model
configurations to optimize both performance and computational efficiency.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import spacy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Query complexity levels for parameter adaptation."""
    SIMPLE = "simple"      # Score: 0.0 - 0.3
    MEDIUM = "medium"      # Score: 0.3 - 0.7  
    COMPLEX = "complex"    # Score: 0.7 - 1.0


@dataclass
class DRAParameters:
    """DRA model parameter configuration."""
    complexity_level: ComplexityLevel
    retrieval_k: int                    # Number of documents to retrieve
    rerank_top_k: int                  # Top-k for reranking
    similarity_threshold: float         # Minimum similarity threshold
    max_context_length: int            # Maximum context window
    temperature: float                 # Generation temperature
    max_tokens: int                    # Maximum output tokens
    lora_rank: int                     # LoRA adaptation rank
    lora_alpha: float                  # LoRA scaling parameter
    
    @classmethod
    def get_default_params(cls, complexity: ComplexityLevel) -> 'DRAParameters':
        """Get default parameters for each complexity level."""
        if complexity == ComplexityLevel.SIMPLE:
            return cls(
                complexity_level=complexity,
                retrieval_k=3,
                rerank_top_k=2,
                similarity_threshold=0.7,
                max_context_length=512,
                temperature=0.3,
                max_tokens=150,
                lora_rank=4,
                lora_alpha=8.0
            )
        elif complexity == ComplexityLevel.MEDIUM:
            return cls(
                complexity_level=complexity,
                retrieval_k=5,
                rerank_top_k=3,
                similarity_threshold=0.6,
                max_context_length=1024,
                temperature=0.5,
                max_tokens=250,
                lora_rank=8,
                lora_alpha=16.0
            )
        else:  # COMPLEX
            return cls(
                complexity_level=complexity,
                retrieval_k=10,
                rerank_top_k=5,
                similarity_threshold=0.5,
                max_context_length=2048,
                temperature=0.7,
                max_tokens=400,
                lora_rank=16,
                lora_alpha=32.0
            )


class QueryComplexityAnalyzer:
    """Analyzes query complexity using multiple linguistic and semantic features."""
    
    def __init__(self):
        """Initialize the complexity analyzer."""
        try:
            # Try to load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic analysis.")
            self.nlp = None
        
        # Domain-specific keywords for oil & gas industry
        self.domain_keywords = {
            'technical': ['drilling', 'production', 'reservoir', 'seismic', 'exploration',
                         'refining', 'pipeline', 'offshore', 'subsea', 'wellhead',
                         'crude', 'hydrocarbon', 'geological', 'geophysical'],
            'financial': ['revenue', 'profit', 'investment', 'capex', 'opex', 'cost',
                         'budget', 'financial', 'earnings', 'cash flow', 'dividend'],
            'regulatory': ['regulation', 'compliance', 'environmental', 'safety',
                          'emission', 'carbon', 'sustainability', 'policy', 'license'],
            'operational': ['operations', 'maintenance', 'efficiency', 'optimization',
                           'performance', 'capacity', 'utilization', 'downtime']
        }
        
        # Complex query patterns
        self.complex_patterns = [
            r'\b(compare|comparison|versus|vs\.?)\b',
            r'\b(analyze|analysis|evaluate|assessment)\b',
            r'\b(trend|trends|over time|temporal)\b',
            r'\b(why|how|what caused|reason|explanation)\b',
            r'\b(predict|forecast|future|projection)\b',
            r'\b(relationship|correlation|impact|effect)\b'
        ]
    
    def analyze_complexity(self, query: str) -> float:
        """
        Analyze query complexity and return a score between 0 and 1.
        
        Args:
            query: Input query string
            
        Returns:
            Complexity score (0.0 = simple, 1.0 = very complex)
        """
        if not query or not query.strip():
            return 0.0
        
        query_lower = query.lower().strip()
        
        # Feature calculations
        features = {
            'length': self._calculate_length_complexity(query_lower),
            'linguistic': self._calculate_linguistic_complexity(query),
            'domain': self._calculate_domain_complexity(query_lower),
            'pattern': self._calculate_pattern_complexity(query_lower),
            'semantic': self._calculate_semantic_complexity(query_lower)
        }
        
        # Weighted combination of features
        weights = {
            'length': 0.15,
            'linguistic': 0.25,
            'domain': 0.20,
            'pattern': 0.25,
            'semantic': 0.15
        }
        
        complexity_score = sum(features[key] * weights[key] for key in features)
        
        # Ensure score is within [0, 1] range
        complexity_score = max(0.0, min(1.0, complexity_score))
        
        logger.debug(f"Query complexity analysis: {features}")
        logger.info(f"Final complexity score: {complexity_score:.3f}")
        
        return complexity_score
    
    def _calculate_length_complexity(self, query: str) -> float:
        """Calculate complexity based on query length."""
        word_count = len(query.split())
        
        if word_count <= 5:
            return 0.1
        elif word_count <= 10:
            return 0.3
        elif word_count <= 20:
            return 0.6
        else:
            return 1.0
    
    def _calculate_linguistic_complexity(self, query: str) -> float:
        """Calculate linguistic complexity using spaCy if available."""
        if not self.nlp:
            # Fallback to simple heuristics
            return self._simple_linguistic_analysis(query)
        
        doc = self.nlp(query)
        
        complexity_factors = 0.0
        total_factors = 4
        
        # Named entities
        if len(doc.ents) > 0:
            complexity_factors += 0.5
        
        # Complex syntactic structures
        complex_deps = ['nsubj', 'dobj', 'prep', 'compound']
        complex_dep_count = sum(1 for token in doc if token.dep_ in complex_deps)
        if complex_dep_count > 2:
            complexity_factors += 0.5
        
        # Multiple clauses
        clause_markers = ['that', 'which', 'when', 'where', 'how', 'why']
        if any(token.text.lower() in clause_markers for token in doc):
            complexity_factors += 0.3
        
        # Complex part-of-speech patterns
        pos_complexity = ['PROPN', 'NUM', 'ADJ']
        pos_count = sum(1 for token in doc if token.pos_ in pos_complexity)
        if pos_count > 2:
            complexity_factors += 0.2
        
        return min(1.0, complexity_factors)
    
    def _simple_linguistic_analysis(self, query: str) -> float:
        """Simple linguistic analysis fallback."""
        complexity = 0.0
        
        # Check for question words
        question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
        if any(word in query.lower() for word in question_words):
            complexity += 0.3
        
        # Check for conjunctions
        conjunctions = ['and', 'or', 'but', 'however', 'therefore', 'because']
        if any(conj in query.lower() for conj in conjunctions):
            complexity += 0.2
        
        # Check for numbers and dates
        if re.search(r'\d+', query):
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _calculate_domain_complexity(self, query: str) -> float:
        """Calculate domain-specific complexity."""
        domain_score = 0.0
        categories_found = 0
        
        for category, keywords in self.domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                categories_found += 1
                
                # Technical terms add more complexity
                if category == 'technical':
                    domain_score += 0.4
                else:
                    domain_score += 0.2
        
        # Multi-domain queries are more complex
        if categories_found > 1:
            domain_score += 0.3
        
        return min(1.0, domain_score)
    
    def _calculate_pattern_complexity(self, query: str) -> float:
        """Calculate complexity based on query patterns."""
        pattern_score = 0.0
        
        for pattern in self.complex_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                pattern_score += 0.2
        
        return min(1.0, pattern_score)
    
    def _calculate_semantic_complexity(self, query: str) -> float:
        """Calculate semantic complexity."""
        complexity = 0.0
        
        # Temporal references
        temporal_words = ['year', 'month', 'quarter', 'period', 'time', 'recent', 'historical']
        if any(word in query for word in temporal_words):
            complexity += 0.3
        
        # Comparative language
        comparative_words = ['better', 'worse', 'higher', 'lower', 'increase', 'decrease']
        if any(word in query for word in comparative_words):
            complexity += 0.3
        
        # Abstract concepts
        abstract_words = ['strategy', 'impact', 'effect', 'influence', 'significance']
        if any(word in query for word in abstract_words):
            complexity += 0.4
        
        return min(1.0, complexity)


class DRAController:
    """
    Dynamic Rank Adaptation Controller
    
    Main controller that orchestrates query complexity analysis and parameter adaptation.
    """
    
    def __init__(self):
        """Initialize the DRA Controller."""
        self.analyzer = QueryComplexityAnalyzer()
        self.parameter_cache = {}
        self.adaptation_history = []
        
        logger.info("DRA Controller initialized")
    
    def adapt_parameters(self, query: str) -> DRAParameters:
        """
        Analyze query and return adapted parameters.
        
        Args:
            query: Input query string
            
        Returns:
            DRAParameters object with optimized settings
        """
        # Check cache first
        if query in self.parameter_cache:
            logger.debug("Using cached parameters")
            return self.parameter_cache[query]
        
        # Analyze query complexity
        complexity_score = self.analyzer.analyze_complexity(query)
        
        # Determine complexity level
        if complexity_score <= 0.3:
            complexity_level = ComplexityLevel.SIMPLE
        elif complexity_score <= 0.7:
            complexity_level = ComplexityLevel.MEDIUM
        else:
            complexity_level = ComplexityLevel.COMPLEX
        
        # Get parameters for complexity level
        parameters = DRAParameters.get_default_params(complexity_level)
        
        # Fine-tune parameters based on exact score
        parameters = self._fine_tune_parameters(parameters, complexity_score)
        
        # Cache result
        self.parameter_cache[query] = parameters
        
        # Track adaptation history
        self.adaptation_history.append({
            'query': query,
            'complexity_score': complexity_score,
            'complexity_level': complexity_level.value,
            'parameters': parameters
        })
        
        logger.info(f"Query adapted: {complexity_level.value} (score: {complexity_score:.3f})")
        
        return parameters
    
    def _fine_tune_parameters(self, params: DRAParameters, score: float) -> DRAParameters:
        """Fine-tune parameters based on exact complexity score."""
        
        # Adjust retrieval parameters based on score
        if params.complexity_level == ComplexityLevel.SIMPLE:
            # For simple queries, be more selective
            params.similarity_threshold = 0.7 + (0.2 * score)
        elif params.complexity_level == ComplexityLevel.MEDIUM:
            # Medium complexity - balanced approach
            params.retrieval_k = int(5 + (score * 3))
            params.similarity_threshold = 0.6 + (0.1 * score)
        else:  # COMPLEX
            # Complex queries need more context
            params.retrieval_k = int(8 + (score * 5))
            params.max_context_length = int(1500 + (score * 1000))
        
        return params
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about parameter adaptations."""
        if not self.adaptation_history:
            return {}
        
        complexity_scores = [entry['complexity_score'] for entry in self.adaptation_history]
        complexity_levels = [entry['complexity_level'] for entry in self.adaptation_history]
        
        stats = {
            'total_adaptations': len(self.adaptation_history),
            'average_complexity': np.mean(complexity_scores),
            'complexity_distribution': {
                level: complexity_levels.count(level) 
                for level in ['simple', 'medium', 'complex']
            },
            'cache_hit_rate': len(self.parameter_cache) / len(self.adaptation_history) if self.adaptation_history else 0
        }
        
        return stats
    
    def clear_cache(self):
        """Clear parameter cache."""
        self.parameter_cache.clear()
        logger.info("Parameter cache cleared")
    
    def reset_history(self):
        """Reset adaptation history."""
        self.adaptation_history.clear()
        logger.info("Adaptation history reset")


if __name__ == "__main__":
    # Example usage
    controller = DRAController()
    
    # Test queries with different complexity levels
    test_queries = [
        "What is oil?",  # Simple
        "How did production change in 2023?",  # Medium
        "Analyze the correlation between drilling efficiency and environmental compliance over the past five years, considering regulatory changes and technological improvements."  # Complex
    ]
    
    print("DRA Controller Test Results:")
    print("=" * 50)
    
    for query in test_queries:
        params = controller.adapt_parameters(query)
        print(f"\nQuery: {query}")
        print(f"Complexity Level: {params.complexity_level.value}")
        print(f"Retrieval K: {params.retrieval_k}")
        print(f"LoRA Rank: {params.lora_rank}")
        print(f"Temperature: {params.temperature}")
    
    print(f"\nAdaptation Statistics:")
    stats = controller.get_adaptation_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")