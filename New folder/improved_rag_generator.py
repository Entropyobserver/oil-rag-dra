import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever


class ImprovedRAGGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.retriever = self.setup_retriever()
        self.generator_model, self.generator_tokenizer = self.setup_generator()
        
    def setup_retriever(self):
        print("Loading retrieval system...")
        embedder = DocumentEmbedder(device=self.device)
        indexer = FAISSIndexer(dimension=768, index_type="IVF")
        indexer.load("models/faiss_index.bin", "models/documents.pkl")
        
        retriever = HybridRetriever(
            embedder=embedder,
            indexer=indexer,
            reranker=None,
            initial_k=50,
            final_k=5
        )
        return retriever
    
    def setup_generator(self):
        print("Loading language model for answer generation...")
        model_name = "google/flan-t5-base"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            if self.device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
            
            print(f"Loaded {model_name} successfully")
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to simple generation...")
            return None, None
    
    def create_generation_prompt(self, question, contexts):
        if not contexts:
            return f"Question: {question}\nAnswer based on the context: No relevant information available."
        
        context_text = ""
        for i, doc in enumerate(contexts[:3]):
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            sentences = text.split('.')[:3]
            clean_text = '. '.join(sentences).strip()
            
            if clean_text:
                context_text += f"[{year}] {clean_text}. "
        
        if len(context_text) > 800:
            context_text = context_text[:800] + "..."
        
        prompt = f"""Answer the question based on the provided context from oil company reports.

Context: {context_text}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer_with_model(self, question, contexts):
        if not self.generator_model or not self.generator_tokenizer:
            return self.fallback_generation(contexts)
        
        prompt = self.create_generation_prompt(question, contexts)
        
        try:
            inputs = self.generator_tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generator_model.generate(
                    **inputs,
                    max_length=200,
                    min_length=30,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    do_sample=True
                )
            
            answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if len(answer.strip()) < 10:
                return self.fallback_generation(contexts)
            
            return answer.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self.fallback_generation(contexts)
    
    def fallback_generation(self, contexts):
        if not contexts:
            return "No relevant information found in the documents."
        
        combined_text = ""
        for doc in contexts[:2]:
            text = doc.get('text', '')
            sentences = text.split('.')[:2]
            combined_text += '. '.join(sentences) + ". "
        
        if len(combined_text) > 300:
            combined_text = combined_text[:300] + "..."
        
        return combined_text.strip() if combined_text.strip() else "Unable to generate answer from available documents."
    
    def search_and_generate(self, question, num_docs=5):
        print(f"\nProcessing: '{question}'")
        
        start_time = time.time()
        docs, scores = self.retriever.retrieve(question, k=num_docs)
        retrieval_time = time.time() - start_time
        
        generation_start = time.time()
        answer = self.generate_answer_with_model(question, docs)
        generation_time = time.time() - generation_start
        
        total_time = retrieval_time + generation_time
        
        result = {
            'question': question,
            'answer': answer,
            'retrieved_docs': len(docs),
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'top_score': scores[0] if scores else 0.0,
            'contexts': docs
        }
        
        return result
    
    def display_result(self, result):
        print(f"\n{'='*60}")
        print(f"QUESTION: {result['question']}")
        print(f"{'='*60}")
        print(f"ANSWER: {result['answer']}")
        print(f"\n{'='*60}")
        print(f"PERFORMANCE:")
        print(f"  Retrieval: {result['retrieval_time']:.3f}s")
        print(f"  Generation: {result['generation_time']:.3f}s") 
        print(f"  Total: {result['total_time']:.3f}s")
        print(f"  Documents: {result['retrieved_docs']}")
        print(f"  Top Score: {result['top_score']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"SOURCE DOCUMENTS:")
        print(f"{'='*60}")
        
        for i, doc in enumerate(result['contexts'][:3]):
            year = doc.get('year', 'N/A')
            lang = doc.get('lang', 'N/A')
            content = doc.get('text', 'No content')[:150] + "..."
            
            print(f"[{i+1}] {year} ({lang}): {content}")
            print("-" * 40)


class ImprovedRAGEvaluator:
    def __init__(self, generator):
        self.generator = generator
    
    def evaluate_sample_questions(self):
        questions = [
            "What safety measures were implemented in 2020?",
            "How did oil production volumes change from 2015 to 2020?",
            "What environmental initiatives did the company take?",
            "What new drilling technologies were developed?",
            "What were the main financial highlights in recent years?"
        ]
        
        results = []
        
        print(f"\n{'='*80}")
        print(f"EVALUATING IMPROVED RAG SYSTEM ON SAMPLE QUESTIONS")
        print(f"{'='*80}")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[Question {i}/{len(questions)}]")
            
            result = self.generator.search_and_generate(question)
            results.append(result)
            
            self.generator.display_result(result)
            
            if i < len(questions):
                input("\nPress Enter for next question...")
        
        return results
    
    def performance_summary(self, results):
        if not results:
            return
        
        avg_retrieval = sum(r['retrieval_time'] for r in results) / len(results)
        avg_generation = sum(r['generation_time'] for r in results) / len(results)
        avg_total = sum(r['total_time'] for r in results) / len(results)
        avg_score = sum(r['top_score'] for r in results) / len(results)
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Questions Processed: {len(results)}")
        print(f"Average Retrieval Time: {avg_retrieval:.3f}s")
        print(f"Average Generation Time: {avg_generation:.3f}s")
        print(f"Average Total Time: {avg_total:.3f}s")
        print(f"Average Relevance Score: {avg_score:.4f}")
        
        print(f"\nGeneration Method: {'FLAN-T5 Model' if self.generator.generator_model else 'Fallback (Simple)'}")


def main():
    print("Improved Oil RAG System with Language Model Generation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        generator = ImprovedRAGGenerator(device=device)
        evaluator = ImprovedRAGEvaluator(generator)
        
        print("\nSystem loaded successfully!")
        print("Choose an option:")
        print("1. Run sample questions evaluation")
        print("2. Interactive mode")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            results = evaluator.evaluate_sample_questions()
            evaluator.performance_summary(results)
        else:
            print("\nInteractive Mode - Enter questions (type 'quit' to exit)")
            print("=" * 50)
            
            while True:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif not question:
                    continue
                
                result = generator.search_and_generate(question)
                generator.display_result(result)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()