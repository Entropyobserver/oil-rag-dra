"""
RAG System Evaluator
Implements comprehensive evaluation for the oil company RAG system.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import sys

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from oil_rag.evaluation.metrics import QAMetrics
from oil_rag.utils.logger import setup_logger


class RAGEvaluator:
    """Evaluator for RAG system performance on QA tasks."""
    
    def __init__(self, test_data_path: str = "data/test/test_qa.jsonl"):
        self.logger = setup_logger('RAGEvaluator')
        self.test_data_path = test_data_path
        self.test_data = self.load_test_data()
    
    def load_test_data(self) -> List[Dict]:
        """Load QA test data from JSONL file."""
        test_data = []
        
        if not Path(self.test_data_path).exists():
            self.logger.error(f"Test data file not found: {self.test_data_path}")
            return []
        
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pair = json.loads(line)
                test_data.append(qa_pair)
        
        self.logger.info(f"Loaded {len(test_data)} QA pairs from {self.test_data_path}")
        return test_data
    
    def evaluate_pipeline(self, pipeline, max_questions: int = None) -> Dict:
        """Evaluate RAG pipeline on test questions."""
        if not self.test_data:
            return {"error": "No test data available"}
        
        self.logger.info("Starting RAG pipeline evaluation")
        
        # Limit questions if specified
        test_questions = self.test_data[:max_questions] if max_questions else self.test_data
        
        predictions = []
        ground_truths = []
        detailed_results = []
        
        start_time = time.time()
        
        for i, qa_pair in enumerate(test_questions):
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            try:
                # Generate answer using RAG pipeline
                result = pipeline.generate_answer(
                    question,
                    max_length=256,
                    use_dra=False,  # Basic evaluation without DRA
                    return_context=True
                )
                
                prediction = result.get('answer', '')
                context_docs = result.get('context', [])
                
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                
                # Store detailed result
                detailed_results.append({
                    'question_id': qa_pair.get('id', f'q_{i}'),
                    'question': question,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'category': qa_pair.get('category', 'unknown'),
                    'year': qa_pair.get('year', 'unknown'),
                    'context_docs': len(context_docs),
                    'retrieval_confidence': result.get('confidence', 0.0)
                })
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(test_questions)} questions")
            
            except Exception as e:
                self.logger.error(f"Error processing question {i}: {e}")
                predictions.append("Error generating answer")
                ground_truths.append(ground_truth)
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        metrics = QAMetrics.compute_batch_metrics(predictions, ground_truths)
        
        # Add timing information
        metrics['evaluation_time'] = evaluation_time
        metrics['questions_per_second'] = len(test_questions) / evaluation_time
        metrics['total_questions'] = len(test_questions)
        
        # Compute category-wise metrics
        category_metrics = self._compute_category_metrics(detailed_results, predictions, ground_truths)
        metrics['category_performance'] = category_metrics
        
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        self.logger.info(f"Overall F1: {metrics['f1']:.3f}, EM: {metrics['exact_match']:.3f}")
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results
        }
    
    def _compute_category_metrics(self, detailed_results: List[Dict], 
                                predictions: List[str], ground_truths: List[str]) -> Dict:
        """Compute metrics per category."""
        categories = {}
        
        for i, result in enumerate(detailed_results):
            category = result['category']
            if category not in categories:
                categories[category] = {'predictions': [], 'ground_truths': []}
            
            if i < len(predictions) and i < len(ground_truths):
                categories[category]['predictions'].append(predictions[i])
                categories[category]['ground_truths'].append(ground_truths[i])
        
        category_metrics = {}
        for category, data in categories.items():
            if data['predictions']:
                metrics = QAMetrics.compute_batch_metrics(
                    data['predictions'], data['ground_truths']
                )
                category_metrics[category] = metrics
        
        return category_metrics
    
    def generate_evaluation_report(self, results: Dict, output_path: str = None) -> str:
        """Generate a human-readable evaluation report."""
        metrics = results['metrics']
        detailed_results = results['detailed_results']
        
        report = []
        report.append("=" * 60)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"  F1 Score:      {metrics['f1']:.3f}")
        report.append(f"  Exact Match:   {metrics['exact_match']:.3f}")
        report.append(f"  BLEU Score:    {metrics.get('bleu', 'N/A')}")
        report.append(f"  ROUGE-L:       {metrics.get('rouge_l', 'N/A')}")
        
        # Timing information
        report.append(f"\nPERFORMANCE METRICS:")
        report.append(f"  Total Questions:     {metrics['total_questions']}")
        report.append(f"  Evaluation Time:     {metrics['evaluation_time']:.2f}s")
        report.append(f"  Questions/Second:    {metrics['questions_per_second']:.2f}")
        
        # Category performance
        if 'category_performance' in metrics:
            report.append(f"\nPERFORMANCE BY CATEGORY:")
            for category, cat_metrics in metrics['category_performance'].items():
                report.append(f"  {category.upper()}:")
                report.append(f"    F1:  {cat_metrics['f1']:.3f}")
                report.append(f"    EM:  {cat_metrics['exact_match']:.3f}")
        
        # Sample results
        report.append(f"\nSAMPLE RESULTS:")
        for i, result in enumerate(detailed_results[:3]):
            report.append(f"\n  Question {i+1}: {result['question']}")
            report.append(f"  Ground Truth: {result['ground_truth'][:100]}...")
            report.append(f"  Prediction:   {result['prediction'][:100]}...")
            report.append(f"  Category:     {result['category']}")
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_text


class AblationStudy:
    """Conduct ablation studies on RAG system components."""
    
    def __init__(self, pipeline, test_data: List[Dict]):
        self.logger = setup_logger('AblationStudy')
        self.pipeline = pipeline
        self.test_data = test_data[:20]  # Use subset for ablation
    
    def run_ablation(self) -> Dict:
        """Run ablation study comparing different configurations."""
        self.logger.info("Starting ablation study")
        
        configurations = [
            {'name': 'baseline', 'use_reranker': False, 'use_dra': False, 'k': 5},
            {'name': 'with_reranker', 'use_reranker': True, 'use_dra': False, 'k': 5},
            {'name': 'more_docs', 'use_reranker': True, 'use_dra': False, 'k': 10},
            {'name': 'with_dra', 'use_reranker': True, 'use_dra': True, 'k': 5},
        ]
        
        results = {}
        
        for config in configurations:
            self.logger.info(f"Testing configuration: {config['name']}")
            
            predictions = []
            ground_truths = []
            
            for qa_pair in self.test_data:
                try:
                    result = self.pipeline.generate_answer(
                        qa_pair['question'],
                        use_dra=config['use_dra'],
                        return_context=True
                    )
                    
                    predictions.append(result.get('answer', ''))
                    ground_truths.append(qa_pair['answer'])
                
                except Exception as e:
                    self.logger.error(f"Error in {config['name']}: {e}")
                    predictions.append("")
                    ground_truths.append(qa_pair['answer'])
            
            # Compute metrics for this configuration
            metrics = QAMetrics.compute_batch_metrics(predictions, ground_truths)
            results[config['name']] = metrics
        
        self.logger.info("Ablation study completed")
        return results
