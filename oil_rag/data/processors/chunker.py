import re
from typing import List, Dict, Optional


class EnhancedChunker:
    def __init__(
        self,
        min_words: int = 30,
        max_words: int = 150,
        overlap_ratio: float = 0.1
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_ratio = overlap_ratio

    def extract_numbers(self, text: str) -> List[str]:
        patterns = [
            r"\d+\.?\d*\s*(?:billion|million|thousand)",
            r"\$\s*\d+\.?\d*",
            r"\d+\.?\d*\s*%",
            r"\d{4}"
        ]
        
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text.lower()))
        return numbers

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > self.max_words and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= self.min_words:
                    chunks.append(chunk_text)
                current_chunk = [sentence]
                current_words = sentence_words
            else:
                current_chunk.append(sentence)
                current_words += sentence_words
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= self.min_words:
                chunks.append(chunk_text)
        
        return chunks

    def chunk_with_metadata(
        self,
        text: str,
        year: int,
        language: str,
        section: str = "Unknown",
        page: int = 0
    ) -> List[Dict]:
        chunks = self.chunk_text(text)
        
        result = []
        for idx, chunk_text in enumerate(chunks):
            result.append({
                "id": f"{language}_{year}_chunk_{idx:06d}",
                "text": chunk_text,
                "year": year,
                "lang": language,
                "section": section,
                "section_normalized": self.normalize_section(section),
                "page": page,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "numbers": self.extract_numbers(chunk_text)
            })
        
        return result

    def normalize_section(self, section: str) -> str:
        normalized = section.lower().strip()
        normalized = re.sub(r"\d+\.?\s*", "", normalized)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized