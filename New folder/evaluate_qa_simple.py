#!/usr/bin/env python3
"""
Simple QA Evaluation Script
Tests the RAG system with generated QA pairs and computes performance metrics.
"""

import sys
import json
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever
from oil_rag.evaluation.evaluate import RAGEvaluator
from oil_rag.evaluation.metrics import QAMetrics
from oil_rag.utils.logger import setup_logger


class SimpleQAEvaluator:
    """Simple QA evaluation without full pipeline (retrieval-only)."""
    
    def __init__(self, index_path="models/faiss_index.bin", 
                 docs_path="models/documents.pkl", device="cpu"):
        self.logger = setup_logger('SimpleQAEvaluator')
        self.device = device
        self.retriever = self._build_retriever(index_path, docs_path)
    
    def _build_retriever(self, index_path, docs_path):
        """Build retriever for QA evaluation."""
        self.logger.info("Building retriever for QA evaluation...")
        
        embedder = DocumentEmbedder(device=self.device)
        indexer = FAISSIndexer(dimension=768, index_type="IVF")
        indexer.load(index_path, docs_path)
        
        retriever = HybridRetriever(
            embedder=embedder,
            indexer=indexer,
            reranker=None,  # Simplified for now
            initial_k=50,
            final_k=5
        )
        
        self.logger.info(f"Retriever ready with {indexer.get_index_size()} documents")
        return retriever
    
    def load_qa_data(self, qa_path="data/test/test_qa.jsonl") -> list:
        """Load QA test data."""
        qa_pairs = []
        
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pair = json.loads(line)
                qa_pairs.append(qa_pair)
        
        self.logger.info(f"Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def simple_answer_generation(self, question: str, retrieved_docs: list) -> str:
        """Generate simple answers from retrieved documents (rule-based)."""
        if not retrieved_docs:
            return "No relevant information found."
        
        # Simple strategy: combine text from top documents
        combined_text = ""
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            text = doc.get('text', '')
            # Take first 2 sentences
            sentences = text.split('.')[:2]
            combined_text += '. '.join(sentences) + ". "
        
        # Limit answer length
        if len(combined_text) > 400:
            combined_text = combined_text[:400] + "..."
        
        return combined_text.strip()
    
    def evaluate_retrieval_qa(self, qa_pairs: list, max_eval: int = 50) -> dict:
        """Evaluate QA performance using retrieval + simple generation."""
        self.logger.info(f"Evaluating QA performance on {min(len(qa_pairs), max_eval)} questions")
        
        predictions = []
        ground_truths = []
        detailed_results = []
        
        start_time = time.time()
        
        for i, qa_pair in enumerate(qa_pairs[:max_eval]):
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            try:
                # Retrieve documents
                docs, scores = self.retriever.retrieve(question, k=5)
                
                # Generate simple answer
                prediction = self.simple_answer_generation(question, docs)
                
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                
                detailed_results.append({
                    'id': qa_pair.get('id', f'q_{i}'),
                    'question': question,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'category': qa_pair.get('category', 'unknown'),
                    'retrieved_docs': len(docs),
                    'top_score': scores[0] if scores else 0.0
                })
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1} questions")
            
            except Exception as e:
                self.logger.error(f"Error processing question {i}: {e}")
                predictions.append("Error occurred")
                ground_truths.append(ground_truth)
        
        # Compute metrics
        eval_time = time.time() - start_time
        metrics = QAMetrics.compute_batch_metrics(predictions, ground_truths)
        
        # Add evaluation metadata
        metrics.update({
            'total_questions': len(predictions),
            'evaluation_time': eval_time,
            'questions_per_second': len(predictions) / eval_time
        })
        
        self.logger.info(f"Evaluation completed in {eval_time:.2f}s")
        self.logger.info(f"F1 Score: {metrics['f1']:.3f}")
        self.logger.info(f"Exact Match: {metrics['exact_match']:.3f}")
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results
        }
    
    def print_sample_results(self, results: dict, num_samples: int = 5):
        """Print sample QA results for inspection."""
        detailed_results = results['detailed_results']
        
        print(f"\n{'='*60}")
        print("SAMPLE QA RESULTS")
        print(f"{'='*60}")
        
        for i, result in enumerate(detailed_results[:num_samples]):
            print(f"\n[Sample {i+1}]")
            print(f"Category: {result['category']}")
            print(f"Question: {result['question']}")
            print(f"Ground Truth: {result['ground_truth'][:150]}...")
            print(f"Prediction: {result['prediction'][:150]}...")
            print(f"Retrieved Docs: {result['retrieved_docs']}")
            print(f"Top Score: {result['top_score']:.4f}")
            print("-" * 40)
    
    def analyze_by_category(self, results: dict):
        """Analyze performance by question category."""
        detailed_results = results['detailed_results']
        
        # Group by category
        categories = {}
        for result in detailed_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'predictions': [], 'ground_truths': []}
            
            categories[cat]['predictions'].append(result['prediction'])
            categories[cat]['ground_truths'].append(result['ground_truth'])
        
        print(f"\n{'='*60}")
        print("PERFORMANCE BY CATEGORY")
        print(f"{'='*60}")
        
        for category, data in categories.items():
            metrics = QAMetrics.compute_batch_metrics(
                data['predictions'], data['ground_truths']
            )
            
            print(f"{category.upper()}: {len(data['predictions'])} questions")
            print(f"  F1 Score: {metrics['f1']:.3f}")
            print(f"  Exact Match: {metrics['exact_match']:.3f}")


def main():
    """Main evaluation function."""
    print("Simple QA Evaluation for Oil RAG System")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "models/faiss_index.bin",
        "models/documents.pkl", 
        "data/test/test_qa.jsonl"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Please run build_index.py and generate_qa_dataset.py first")
        return
    
    try:
        # Initialize evaluator
        evaluator = SimpleQAEvaluator(device="cpu")
        
        # Load QA data
        qa_pairs = evaluator.load_qa_data()
        
        if not qa_pairs:
            print("No QA pairs found!")
            return
        
        # Run evaluation
        results = evaluator.evaluate_retrieval_qa(qa_pairs, max_eval=50)
        
        # Print overall results
        metrics = results['metrics']
        print(f"\nOVERALL RESULTS:")
        print(f"Questions Evaluated: {metrics['total_questions']}")
        print(f"F1 Score: {metrics['f1']:.3f} (±{metrics['f1_std']:.3f})")
        print(f"Exact Match: {metrics['exact_match']:.3f} (±{metrics['em_std']:.3f})")
        print(f"Evaluation Time: {metrics['evaluation_time']:.2f}s")
        print(f"Speed: {metrics['questions_per_second']:.2f} questions/sec")
        
        # Show sample results
        evaluator.print_sample_results(results)
        
        # Analyze by category
        evaluator.analyze_by_category(results)
        
        # Save results
        output_path = "evaluation_results/simple_qa_results.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()