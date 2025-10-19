import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from oil_rag.retrieval.retriever import BilingualRetriever


def test_retriever_initialization():
    retriever = BilingualRetriever()
    assert retriever.top_k == 20
    assert retriever.model is not None


def test_add_documents():
    retriever = BilingualRetriever()
    docs = [
        {"id": "1", "text": "Oil and gas production", "year": 2024},
        {"id": "2", "text": "Renewable energy investments", "year": 2024}
    ]
    retriever.add_documents(docs, "en")
    assert "en" in retriever.index
    assert len(retriever.chunks["en"]) == 2


def test_retrieve():
    retriever = BilingualRetriever(similarity_threshold=0.0)
    docs = [
        {"id": "1", "text": "Oil and gas production increased", "year": 2024},
        {"id": "2", "text": "Renewable energy investments grew", "year": 2024}
    ]
    retriever.add_documents(docs, "en")
    
    results = retriever.retrieve("oil production", "en", top_k=1)
    assert len(results) > 0
    assert "score" in results[0]


def test_retrieve_bilingual():
    retriever = BilingualRetriever(similarity_threshold=0.0)
    en_docs = [{"id": "1", "text": "Oil production", "year": 2024}]
    no_docs = [{"id": "1", "text": "Oljeproduksjon", "year": 2024}]
    
    retriever.add_documents(en_docs, "en")
    retriever.add_documents(no_docs, "no")
    
    results = retriever.retrieve_bilingual("oil", "en", "no", 5)
    assert "en" in results
    assert "no" in results
