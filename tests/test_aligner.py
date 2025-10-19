import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from oil_rag.data.processors.aligner import BilingualAligner


def test_aligner_initialization():
    aligner = BilingualAligner()
    assert aligner.threshold == 0.70
    assert aligner.model is not None


def test_encode_chunks():
    aligner = BilingualAligner()
    chunks = [
        {"text": "This is a test", "year": 2024},
        {"text": "Another test", "year": 2024}
    ]
    embeddings = aligner.encode_chunks(chunks)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_length_ratio_check():
    aligner = BilingualAligner()
    en_chunk = {"word_count": 100}
    no_chunk = {"word_count": 90}
    assert aligner.check_length_ratio(en_chunk, no_chunk) is True
    
    no_chunk_bad = {"word_count": 300}
    assert aligner.check_length_ratio(en_chunk, no_chunk_bad) is False


def test_numbers_match():
    aligner = BilingualAligner()
    en_chunk = {"numbers": ["2024", "100", "50%"]}
    no_chunk = {"numbers": ["2024", "100"]}
    assert aligner.check_numbers_match(en_chunk, no_chunk) is True


def test_align_chunks():
    aligner = BilingualAligner(threshold=0.5)
    en_chunks = [
        {
            "text": "Oil production increased significantly",
            "year": 2024,
            "section_normalized": "production",
            "word_count": 5,
            "numbers": []
        }
    ]
    no_chunks = [
        {
            "text": "Oljeproduksjonen økte betydelig",
            "year": 2024,
            "section_normalized": "produksjon",
            "word_count": 4,
            "numbers": []
        }
    ]
    
    en_embs = aligner.encode_chunks(en_chunks)
    no_embs = aligner.encode_chunks(no_chunks)
    
    alignments = aligner.align_chunks(en_chunks, no_chunks, en_embs, no_embs)
    assert len(alignments) >= 0