import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from oil_rag.data.processors.chunker import EnhancedChunker
from oil_rag.data.processors.section_extractor import SectionExtractor


def test_chunker_initialization():
    chunker = EnhancedChunker()
    assert chunker.min_words == 30
    assert chunker.max_words == 150


def test_extract_numbers():
    chunker = EnhancedChunker()
    text = "Revenue increased to $100 million in 2024, up 50%"
    numbers = chunker.extract_numbers(text)
    assert len(numbers) > 0


def test_chunk_text():
    chunker = EnhancedChunker(min_words=5, max_words=20)
    text = "This is a test sentence. " * 10
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0


def test_section_extractor():
    extractor = SectionExtractor()
    assert extractor.is_section_header("INTRODUCTION") is True
    assert extractor.is_section_header("This is regular text") is False


def test_normalize_section():
    chunker = EnhancedChunker()
    assert chunker.normalize_section("1. Introduction") == "introduction"
    assert chunker.normalize_section("CHAPTER 2: Results") == "chapter results"