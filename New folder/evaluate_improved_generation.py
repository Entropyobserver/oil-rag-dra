import json
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.evaluation.metrics import QAMetrics
from smart_answer_generator import SmartAnswerGenerator


class AnswerQualityEvaluator:
    def __init__(self):
        self.generator = SmartAnswerGenerator()
        self.test_questions = self.load_test_questions()
    
    def load_test_questions(self):
        qa_pairs = []
        qa_file = "data/test/test_qa.jsonl"
        
        if not Path(qa_file).exists():
            print(f"Test file {qa_file} not found, using sample questions")
            return [
                {
                    "question": "What safety measures were implemented in 2020?",
                    "answer": "Safety measures included blowout preventers, regular inspections, and safety management systems.",
                    "category": "safety"
                },
                {
                    "question": "How did oil production change over the years?",
                    "answer": "Oil production volumes varied based on market conditions and operational efficiency improvements.",
                    "category": "production"
                },
                {
                    "question": "What were the financial performance highlights?",
                    "answer": "Financial highlights included revenue growth, cost optimization, and improved profitability.",
                    "category": "financial"
                },
                {
                    "question": "What environmental initiatives were taken?",
                    "answer": "Environmental initiatives focused on emission reduction, renewable energy, and sustainability programs.",
                    "category": "environment"
                },
                {
                    "question": "What new technologies were developed?",
                    "answer": "New technologies included digital solutions, advanced drilling techniques, and data analytics.",
                    "category": "technology"
                }
            ]
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pair = json.loads(line)
                qa_pairs.append(qa_pair)
        
        return qa_pairs[:20]  # Use first 20 for evaluation
    
    def evaluate_improved_generation(self):
        print("Evaluating Improved Answer Generation System")
        print("=" * 60)
        
        predictions = []
        ground_truths = []
        detailed_results = []
        
        total_start_time = time.time()
        
        for i, qa_pair in enumerate(self.test_questions):
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            category = qa_pair.get('category', 'general')
            
            print(f"\n[{i+1}/{len(self.test_questions)}] Processing: {category}")
            
            result = self.generator.search_and_answer(question)
            prediction = result['answer']
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            
            detailed_results.append({
                'question_id': qa_pair.get('id', f'q_{i}'),
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'category': category,
                'retrieval_time': result['retrieval_time'],
                'generation_time': result['generation_time'],
                'total_time': result['total_time'],
                'docs_found': result['docs_found'],
                'top_score': result['top_score']
            })
            
            print(f"Generated: {prediction[:100]}...")
        
        total_evaluation_time = time.time() - total_start_time
        
        metrics = QAMetrics.compute_batch_metrics(predictions, ground_truths)
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'evaluation_time': total_evaluation_time,
            'questions_processed': len(self.test_questions)
        }
    
    def compare_with_baseline(self):
        baseline_f1 = 0.055
        baseline_em = 0.000
        baseline_speed = 35.21
        
        results = self.evaluate_improved_generation()
        metrics = results['metrics']
        
        improved_f1 = metrics['f1']
        improved_em = metrics['exact_match']
        improved_speed = results['questions_processed'] / results['evaluation_time']
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE COMPARISON: BASELINE vs IMPROVED")
        print(f"{'='*80}")
        
        print(f"{'Metric':<20} {'Baseline':<12} {'Improved':<12} {'Change':<15}")
        print(f"{'-'*60}")
        print(f"{'F1 Score':<20} {baseline_f1:<12.3f} {improved_f1:<12.3f} {'+' if improved_f1 > baseline_f1 else ''}{((improved_f1 - baseline_f1) / baseline_f1 * 100):>6.1f}%")
        print(f"{'Exact Match':<20} {baseline_em:<12.3f} {improved_em:<12.3f} {'+' if improved_em > baseline_em else ''}{((improved_em - baseline_em) if baseline_em > 0 else improved_em):>6.3f}")
        print(f"{'Speed (q/s)':<20} {baseline_speed:<12.1f} {improved_speed:<12.1f} {'+' if improved_speed > baseline_speed else ''}{((improved_speed - baseline_speed) / baseline_speed * 100):>6.1f}%")
        
        print(f"\n{'='*80}")
        print(f"QUALITY IMPROVEMENTS:")
        print(f"{'='*80}")
        print(f"✓ Topic-specific answer formatting")
        print(f"✓ Keyword-based relevance scoring")
        print(f"✓ Multi-document information synthesis")
        print(f"✓ Year-based information organization")
        print(f"✓ Context-aware sentence extraction")
        
        improvement_summary = {
            'f1_improvement': ((improved_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0,
            'em_improvement': improved_em - baseline_em,
            'speed_change': ((improved_speed - baseline_speed) / baseline_speed * 100),
            'qualitative_improvements': [
                'Topic-specific formatting',
                'Better keyword matching',
                'Multi-document synthesis',
                'Temporal organization',
                'Context-aware extraction'
            ]
        }
        
        return improvement_summary, results
    
    def analyze_by_category(self, results):
        detailed_results = results['detailed_results']
        
        categories = {}
        for result in detailed_results:
            category = result['category']
            if category not in categories:
                categories[category] = {
                    'predictions': [],
                    'ground_truths': [],
                    'response_times': [],
                    'doc_counts': []
                }
            
            categories[category]['predictions'].append(result['prediction'])
            categories[category]['ground_truths'].append(result['ground_truth'])
            categories[category]['response_times'].append(result['total_time'])
            categories[category]['doc_counts'].append(result['docs_found'])
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE BY CATEGORY:")
        print(f"{'='*80}")
        
        for category, data in categories.items():
            if data['predictions']:
                metrics = QAMetrics.compute_batch_metrics(
                    data['predictions'], data['ground_truths']
                )
                
                avg_time = sum(data['response_times']) / len(data['response_times'])
                avg_docs = sum(data['doc_counts']) / len(data['doc_counts'])
                
                print(f"\n{category.upper()}:")
                print(f"  Questions: {len(data['predictions'])}")
                print(f"  F1 Score: {metrics['f1']:.3f}")
                print(f"  Exact Match: {metrics['exact_match']:.3f}")
                print(f"  Avg Response Time: {avg_time:.3f}s")
                print(f"  Avg Documents Found: {avg_docs:.1f}")


def main():
    print("Answer Quality Evaluation - Improved vs Baseline")
    print("=" * 60)
    
    evaluator = AnswerQualityEvaluator()
    
    try:
        print("Running improved answer generation evaluation...")
        improvement_summary, results = evaluator.compare_with_baseline()
        
        print(f"\n{'='*80}")
        print(f"SUMMARY:")
        print(f"{'='*80}")
        
        if improvement_summary['f1_improvement'] > 0:
            print(f"✅ F1 Score improved by {improvement_summary['f1_improvement']:.1f}%")
        else:
            print(f"⚠️  F1 Score needs further improvement")
        
        if improvement_summary['em_improvement'] > 0:
            print(f"✅ Exact Match improved by {improvement_summary['em_improvement']:.3f}")
        else:
            print(f"⚠️  Exact Match needs improvement")
        
        if improvement_summary['speed_change'] > 0:
            print(f"✅ Speed improved by {improvement_summary['speed_change']:.1f}%")
        else:
            print(f"⚠️  Speed decreased by {abs(improvement_summary['speed_change']):.1f}%")
        
        print(f"\n🎯 Next Steps for Further Improvement:")
        print(f"   1. Integrate actual language models (T5, BART)")
        print(f"   2. Add query expansion and reranking")
        print(f"   3. Implement semantic similarity matching")
        print(f"   4. Add few-shot prompting examples")
        
        evaluator.analyze_by_category(results)
        
        output_file = "evaluation_results/improved_generation_results.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'improvement_summary': improvement_summary,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()