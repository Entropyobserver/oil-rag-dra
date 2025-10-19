#!/usr/bin/env python3
"""
Interactive Oil RAG System - Command Line Interface
Simple terminal-based interface for testing the RAG system
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever

class InteractiveRAG:
    def __init__(self):
        print("Loading Oil RAG System...")
        self.retriever = self.setup_retriever()
        print("System ready!")
    
    def setup_retriever(self):
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
        return retriever
    
    def generate_answer(self, docs):
        if not docs:
            return "No relevant information found"
        
        combined_text = ""
        for doc in docs[:3]:
            text = doc.get('text', '')
            sentences = text.split('.')[:2]
            combined_text += '. '.join(sentences) + ". "
        
        if len(combined_text) > 400:
            combined_text = combined_text[:400] + "..."
        
        return combined_text.strip()
    
    def search_and_answer(self, query):
        print(f"\nSearching for: '{query}'")
        
        start_time = time.time()
        docs, scores = self.retriever.retrieve(query, k=5)
        search_time = time.time() - start_time
        
        print(f"Search completed in {search_time:.3f} seconds")
        print(f"Found {len(docs)} relevant documents")
        
        if docs:
            answer = self.generate_answer(docs)
            
            print("\n" + "="*60)
            print("GENERATED ANSWER:")
            print("="*60)
            print(answer)
            
            print("\n" + "="*60)
            print("TOP RETRIEVED DOCUMENTS:")
            print("="*60)
            
            for i, (doc, score) in enumerate(zip(docs, scores)):
                print(f"\n[Document {i+1}] Score: {score:.4f}")
                print(f"Year: {doc.get('year', 'N/A')} | Language: {doc.get('lang', 'N/A')}")
                
                content = doc.get('text', 'No content')
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"Content: {content}")
                print("-" * 40)
        else:
            print("No relevant documents found")
    
    def run_samples(self):
        sample_queries = [
            "What safety measures were implemented in 2020?",
            "How did oil production change over the years?",
            "What environmental initiatives were taken?",
            "What new technologies were developed?",
            "What were the financial performance highlights?"
        ]
        
        print("\n" + "="*80)
        print("RUNNING SAMPLE QUERIES")
        print("="*80)
        
        for query in sample_queries:
            self.search_and_answer(query)
            input("\nPress Enter to continue to next sample...")
    
    def interactive_mode(self):
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("Enter your questions (type 'quit' to exit, 'samples' for sample queries)")
        print("="*80)
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() == 'samples':
                self.run_samples()
                continue
            elif not query:
                continue
            
            self.search_and_answer(query)

def main():
    print("Oil Company Reports RAG System - Interactive CLI")
    print("=" * 50)
    
    try:
        rag = InteractiveRAG()
        
        print("\nOptions:")
        print("1. Interactive mode - ask custom questions")
        print("2. Sample queries - see predefined examples")
        
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == '2':
            rag.run_samples()
        else:
            rag.interactive_mode()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()