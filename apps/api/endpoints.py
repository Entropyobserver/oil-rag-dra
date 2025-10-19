from flask import Flask, request, jsonify
import sys
from pathlib import Path
import time

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever

app = Flask(__name__)

class RAGService:
    def __init__(self):
        self.retriever = self.setup_retriever()
    
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
    
    def generate_simple_answer(self, query, docs):
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

rag_service = RAGService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Oil RAG API"})

@app.route('/search', methods=['POST'])
def search_documents():
    try:
        data = request.get_json()
        query = data.get('query', '')
        num_results = data.get('num_results', 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        start_time = time.time()
        docs, scores = rag_service.retriever.retrieve(query, k=num_results)
        search_time = time.time() - start_time
        
        results = []
        for doc, score in zip(docs, scores):
            results.append({
                "content": doc.get('text', '')[:300] + "..." if len(doc.get('text', '')) > 300 else doc.get('text', ''),
                "year": doc.get('year'),
                "language": doc.get('lang'),
                "section": doc.get('section'),
                "relevance_score": float(score)
            })
        
        return jsonify({
            "query": query,
            "results": results,
            "search_time": search_time,
            "total_results": len(results)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        start_time = time.time()
        docs, scores = rag_service.retriever.retrieve(query, k=5)
        answer = rag_service.generate_simple_answer(query, docs)
        total_time = time.time() - start_time
        
        return jsonify({
            "question": query,
            "answer": answer,
            "response_time": total_time,
            "sources_count": len(docs),
            "top_relevance_score": float(scores[0]) if scores else 0
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Oil Company RAG API",
        "endpoints": {
            "health": "GET /health",
            "search": "POST /search",
            "ask": "POST /ask"
        },
        "usage": {
            "search": "POST /search with JSON: {'query': 'your question', 'num_results': 5}",
            "ask": "POST /ask with JSON: {'query': 'your question'}"
        }
    })

if __name__ == '__main__':
    print("Starting Oil RAG API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
