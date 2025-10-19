import sys
from pathlib import Path
import time
import re

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever


class SmartAnswerGenerator:
    def __init__(self):
        self.retriever = self.setup_retriever()
        
    def setup_retriever(self):
        print("Loading retrieval system...")
        embedder = DocumentEmbedder(device="cpu")
        indexer = FAISSIndexer(dimension=768, index_type="IVF")
        indexer.load("models/faiss_index.bin", "models/documents.pkl")
        
        retriever = HybridRetriever(
            embedder=embedder,
            indexer=indexer,
            reranker=None,
            initial_k=50,
            final_k=5
        )
        print("Retrieval system loaded successfully")
        return retriever
    
    def extract_key_information(self, text, question):
        question_lower = question.lower()
        text_lower = text.lower()
        
        sentences = text.split('.')
        relevant_sentences = []
        
        question_keywords = self.extract_question_keywords(question_lower)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            
            for keyword in question_keywords:
                if keyword in sentence_lower:
                    score += 2
                    
            if any(word in sentence_lower for word in ['million', 'billion', 'percent', '%', 'increased', 'decreased']):
                score += 1
                
            if any(word in sentence_lower for word in ['safety', 'environmental', 'production', 'financial']):
                score += 1
                
            if score >= 2:
                relevant_sentences.append((sentence, score))
        
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in relevant_sentences[:3]]
    
    def extract_question_keywords(self, question):
        stop_words = {'what', 'how', 'when', 'where', 'why', 'did', 'was', 'were', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    def format_answer_by_topic(self, question, docs):
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['safety', 'accident', 'incident', 'risk']):
            return self.format_safety_answer(docs)
        elif any(word in question_lower for word in ['production', 'volume', 'barrel', 'oil', 'gas']):
            return self.format_production_answer(docs)
        elif any(word in question_lower for word in ['financial', 'revenue', 'profit', 'earnings', 'cost']):
            return self.format_financial_answer(docs)
        elif any(word in question_lower for word in ['environment', 'climate', 'emission', 'carbon', 'sustainability']):
            return self.format_environmental_answer(docs)
        elif any(word in question_lower for word in ['technology', 'digital', 'innovation', 'development']):
            return self.format_technology_answer(docs)
        else:
            return self.format_general_answer(question, docs)
    
    def format_safety_answer(self, docs):
        safety_info = []
        years = set()
        
        for doc in docs[:3]:
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            relevant = self.extract_key_information(text, "safety measures incidents")
            if relevant:
                safety_info.extend(relevant)
                years.add(str(year))
        
        if not safety_info:
            return "No specific safety information found in the retrieved documents."
        
        year_range = f"({min(years)}-{max(years)})" if len(years) > 1 else f"({list(years)[0]})" if years else ""
        
        answer = f"Safety measures and performance {year_range}: "
        answer += " ".join(safety_info[:2])
        
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    
    def format_production_answer(self, docs):
        production_info = []
        years_data = {}
        
        for doc in docs[:3]:
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            relevant = self.extract_key_information(text, "production volume barrels oil gas")
            if relevant:
                years_data[year] = relevant
        
        if not years_data:
            return "No specific production information found in the retrieved documents."
        
        answer = "Production information: "
        for year, info in sorted(years_data.items()):
            answer += f"In {year}, {info[0]} " if info else ""
        
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    
    def format_financial_answer(self, docs):
        financial_info = []
        
        for doc in docs[:3]:
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            relevant = self.extract_key_information(text, "financial revenue profit earnings performance")
            if relevant:
                financial_info.extend([f"({year}) {r}" for r in relevant[:1]])
        
        if not financial_info:
            return "No specific financial information found in the retrieved documents."
        
        answer = "Financial performance: " + " ".join(financial_info[:3])
        
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    
    def format_environmental_answer(self, docs):
        env_info = []
        
        for doc in docs[:3]:
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            relevant = self.extract_key_information(text, "environmental climate carbon emission sustainability")
            if relevant:
                env_info.extend(relevant[:1])
        
        if not env_info:
            return "No specific environmental information found in the retrieved documents."
        
        answer = "Environmental initiatives and impact: " + " ".join(env_info[:2])
        
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    
    def format_technology_answer(self, docs):
        tech_info = []
        
        for doc in docs[:3]:
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            relevant = self.extract_key_information(text, "technology digital innovation development research")
            if relevant:
                tech_info.extend([f"In {year}: {r}" for r in relevant[:1]])
        
        if not tech_info:
            return "No specific technology information found in the retrieved documents."
        
        answer = "Technology and innovation: " + " ".join(tech_info[:2])
        
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    
    def format_general_answer(self, question, docs):
        all_relevant = []
        
        for doc in docs[:3]:
            year = doc.get('year', 'Unknown')
            text = doc.get('text', '')
            
            relevant = self.extract_key_information(text, question)
            if relevant:
                all_relevant.extend([f"({year}) {r}" for r in relevant[:1]])
        
        if not all_relevant:
            return "Based on the retrieved documents, specific information related to your question was not found."
        
        answer = " ".join(all_relevant[:3])
        
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    
    def generate_answer(self, question, docs):
        if not docs:
            return "No relevant documents found to answer your question."
        
        return self.format_answer_by_topic(question, docs)
    
    def search_and_answer(self, question):
        print(f"\nProcessing: '{question}'")
        
        start_time = time.time()
        docs, scores = self.retriever.retrieve(question, k=5)
        retrieval_time = time.time() - start_time
        
        generation_start = time.time()
        answer = self.generate_answer(question, docs)
        generation_time = time.time() - generation_start
        
        return {
            'question': question,
            'answer': answer,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time,
            'docs_found': len(docs),
            'top_score': scores[0] if scores else 0.0,
            'sources': docs
        }
    
    def display_result(self, result):
        print(f"\n{'='*60}")
        print(f"QUESTION: {result['question']}")
        print(f"{'='*60}")
        print(f"IMPROVED ANSWER:")
        print(f"{result['answer']}")
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE METRICS:")
        print(f"{'='*60}")
        print(f"Retrieval Time: {result['retrieval_time']:.3f}s")
        print(f"Generation Time: {result['generation_time']:.3f}s")
        print(f"Total Time: {result['total_time']:.3f}s")
        print(f"Documents Found: {result['docs_found']}")
        print(f"Top Relevance: {result['top_score']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"SOURCE DOCUMENTS:")
        print(f"{'='*60}")
        for i, doc in enumerate(result['sources'][:3]):
            print(f"[{i+1}] Year: {doc.get('year', 'N/A')} | Lang: {doc.get('lang', 'N/A')}")
            content = doc.get('text', 'No content')[:120] + "..."
            print(f"    {content}")
            print("-" * 40)


def main():
    print("Smart Answer Generator for Oil RAG System")
    print("=" * 50)
    
    try:
        generator = SmartAnswerGenerator()
        
        sample_questions = [
            "What safety measures were implemented in 2020?",
            "How did oil production volumes change over the years?",
            "What were the financial performance highlights?",
            "What environmental initiatives were taken?",
            "What new technologies were developed?"
        ]
        
        print("\nChoose mode:")
        print("1. Test sample questions")
        print("2. Interactive mode")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print(f"\nTesting {len(sample_questions)} sample questions...")
            
            results = []
            for i, question in enumerate(sample_questions, 1):
                print(f"\n[Question {i}/{len(sample_questions)}]")
                result = generator.search_and_answer(question)
                results.append(result)
                generator.display_result(result)
                
                if i < len(sample_questions):
                    input("\nPress Enter for next question...")
            
            avg_total_time = sum(r['total_time'] for r in results) / len(results)
            avg_generation_time = sum(r['generation_time'] for r in results) / len(results)
            
            print(f"\n{'='*60}")
            print(f"OVERALL PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            print(f"Questions Processed: {len(results)}")
            print(f"Average Total Time: {avg_total_time:.3f}s")
            print(f"Average Generation Time: {avg_generation_time:.3f}s")
            print(f"Answer Quality: Improved with topic-specific formatting")
            
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
                
                result = generator.search_and_answer(question)
                generator.display_result(result)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()