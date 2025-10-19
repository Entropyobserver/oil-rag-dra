#!/usr/bin/env python3
"""
Simple RAG System Tester
Tests the oil company reports RAG pipeline with pre-built FAISS index.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever
from oil_rag.utils.logger import setup_logger


class SimpleRAGTester:
    """Simple RAG system tester for oil company reports."""
    
    def __init__(self, index_path="models/faiss_index.bin", 
                 docs_path="models/documents.pkl", device="cpu"):
        self.logger = setup_logger('RAGTester')
        self.device = device
        self.retriever = self._build_retriever(index_path, docs_path)
    
    def _build_retriever(self, index_path, docs_path):
        """Build and return retriever with loaded index."""
        self.logger.info("Building retriever...")
        
        # Initialize components
        embedder = DocumentEmbedder(device=self.device)
        indexer = FAISSIndexer(dimension=768, index_type="IVF")
        
        # Load pre-built index
        indexer.load(index_path, docs_path)
        self.logger.info(f"Loaded index with {indexer.get_index_size()} documents")
        
        # Create retriever (without reranker for simplicity)
        retriever = HybridRetriever(
            embedder=embedder,
            indexer=indexer,
            reranker=None,
            initial_k=50,
            final_k=5
        )
        
        return retriever
    
    def search(self, query, top_k=3):
        """Search for relevant documents given a query."""
        self.logger.info(f"Searching for: '{query}'")
        
        try:
            docs, scores = self.retriever.retrieve(query, k=top_k)
            return docs, scores
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return [], []
    
    def display_results(self, query, docs, scores):
        """Display search results in a readable format."""
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        if not docs:
            print("No results found.")
            return
        
        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            print(f"\n[Result {i}] (Score: {score:.4f})")
            print(f"Year: {doc.get('year', 'N/A')}")
            print(f"Language: {doc.get('lang', 'N/A')}")
            print(f"Section: {doc.get('section', 'N/A')}")
            print(f"Text: {doc.get('text', 'No text')[:200]}...")
            print("-" * 40)
    
    def run_sample_queries(self):
        """Run a set of sample queries for testing."""
        sample_queries = [
            "What are the safety measures for offshore drilling?",
            "Carbon capture and storage technology",
            "Financial performance and oil prices",
            "Environmental impact and sustainability",
            "Natural gas production volumes"
        ]
        
        print(f"\n{'='*80}")
        print("RUNNING SAMPLE QUERIES")
        print(f"{'='*80}")
        
        for query in sample_queries:
            docs, scores = self.search(query)
            self.display_results(query, docs, scores)
    
    def interactive_mode(self):
        """Interactive query mode."""
        print(f"\n{'='*80}")
        print("INTERACTIVE MODE - Enter your queries (type 'quit' to exit)")
        print(f"{'='*80}")
        
        while True:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            docs, scores = self.search(query)
            self.display_results(query, docs, scores)


def main():
    """Main function to run the RAG tester."""
    print("Oil Company Reports RAG System Tester")
    print("=" * 40)
    
    # Check if index files exist
    index_path = "models/faiss_index.bin"
    docs_path = "models/documents.pkl"
    
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        print(f"ERROR: Index files not found!")
        print(f"Please ensure these files exist:")
        print(f"  - {index_path}")
        print(f"  - {docs_path}")
        return
    
    try:
        # Initialize tester
        tester = SimpleRAGTester(index_path, docs_path, device="cpu")
        
        # Run sample queries
        tester.run_sample_queries()
        
        # Enter interactive mode
        tester.interactive_mode()
        
    except Exception as e:
        print(f"ERROR: {e}")
        return


if __name__ == "__main__":
    main()