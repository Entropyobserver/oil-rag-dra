import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
from config.config import config


def generate_test_queries() -> list:
    
    test_queries = [
        {
            "query": "oil production statistics 2020",
            "relevant_doc_ids": [],
            "difficulty": "easy",
            "category": "production"
        },
        {
            "query": "environmental protection initiatives",
            "relevant_doc_ids": [],
            "difficulty": "medium",
            "category": "environment"
        },
        {
            "query": "safety measures offshore operations",
            "relevant_doc_ids": [],
            "difficulty": "medium",
            "category": "safety"
        },
        {
            "query": "renewable energy investments strategy",
            "relevant_doc_ids": [],
            "difficulty": "hard",
            "category": "strategy"
        },
        {
            "query": "carbon capture storage technology",
            "relevant_doc_ids": [],
            "difficulty": "hard",
            "category": "technology"
        },
        {
            "query": "financial performance 2019",
            "relevant_doc_ids": [],
            "difficulty": "easy",
            "category": "financial"
        },
        {
            "query": "subsea equipment maintenance",
            "relevant_doc_ids": [],
            "difficulty": "medium",
            "category": "operations"
        },
        {
            "query": "compare production volumes 2018 versus 2020",
            "relevant_doc_ids": [],
            "difficulty": "hard",
            "category": "analysis"
        },
        {
            "query": "drilling operations North Sea",
            "relevant_doc_ids": [],
            "difficulty": "medium",
            "category": "operations"
        },
        {
            "query": "cost reduction programs",
            "relevant_doc_ids": [],
            "difficulty": "medium",
            "category": "financial"
        }
    ]
    
    return test_queries


def annotate_relevant_docs(test_queries: list, retriever) -> list:
    
    print("Annotating queries with relevant documents...")
    
    for i, query_data in enumerate(test_queries):
        query = query_data["query"]
        
        documents, scores = retriever.retrieve(query, k=20)
        
        print(f"\n--- Query {i+1}: {query} ---")
        
        relevant_ids = []
        for j, (doc, score) in enumerate(zip(documents[:5], scores[:5])):
            print(f"\n{j+1}. Score: {score:.4f}")
            print(f"   {doc.get('text', '')[:150]}...")
            
            response = input("   Relevant? (y/n/skip): ").strip().lower()
            
            if response == 'y':
                relevant_ids.append(doc.get("id", ""))
            elif response == 'skip':
                break
        
        query_data["relevant_doc_ids"] = relevant_ids
    
    return test_queries


def save_test_queries(test_queries: list, output_path: Path):
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_queries, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved test queries to: {output_path}")


def main():
    test_queries = generate_test_queries()
    
    print(f"Generated {len(test_queries)} test queries")
    
    annotate = input("\nAnnotate with relevant docs? (y/n): ").strip().lower()
    
    if annotate == 'y':
        from oil_rag.core.system_builder import get_retriever
        
        retriever = get_retriever(mode="dense", use_reranker=False)
        test_queries = annotate_relevant_docs(test_queries, retriever)
    
    output_path = config.paths.results_dir / "test_queries.json"
    save_test_queries(test_queries, output_path)


if __name__ == "__main__":
    main()