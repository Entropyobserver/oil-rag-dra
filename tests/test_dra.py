import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from oil_rag.core.dra_controller import DRAController
from oil_rag.retrieval.retriever import BilingualRetriever


def test_dra_initialization():
    retriever = BilingualRetriever()
    dra = DRAController(retriever)
    assert dra.max_iterations == 3
    assert dra.confidence_threshold == 0.8


def test_compute_confidence():
    retriever = BilingualRetriever()
    dra = DRAController(retriever)
    
    chunks = [
        {"score": 0.9},
        {"score": 0.85},
        {"score": 0.8}
    ]
    confidence = dra.compute_confidence(chunks)
    assert 0.0 <= confidence <= 1.0


def test_should_continue_retrieval():
    retriever = BilingualRetriever()
    dra = DRAController(retriever, max_iterations=2)
    
    from oil_rag.core.dra_controller import RetrievalContext
    
    context = RetrievalContext(
        query="test",
        language="en",
        retrieved_chunks=[],
        confidence=0.5,
        iteration=0
    )
    assert dra.should_continue_retrieval(context) is True
    
    context.iteration = 2
    assert dra.should_continue_retrieval(context) is False


def test_retrieve():
    retriever = BilingualRetriever(similarity_threshold=0.0)
    docs = [{"id": "1", "text": "Oil production data", "year": 2024}]
    retriever.add_documents(docs, "en")
    
    dra = DRAController(retriever, max_iterations=1)
    context = dra.retrieve("oil", "en", 5)
    
    assert context is not None
    assert context.query == "oil"
    assert len(dra.history) > 0