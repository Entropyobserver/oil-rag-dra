import re
from typing import List, Dict
from src.data.section_extractor import SectionExtractor


class EnhancedChunker:
    def __init__(self, min_words: int = 30, max_words: int = 150):
        self.min_words = min_words
        self.max_words = max_words
        self.section_extractor = SectionExtractor()
    
    def extract_numbers(self, text: str) -> List[str]:
        patterns = [
            r'\d+\.?\d*\s*(?:billion|million|thousand)',
            r'\$\s*\d+\.?\d*',
            r'\d+\.?\d*\s*%',
            r'\d{4}'
        ]
        
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text.lower()))
        return numbers
    
    def chunk_with_metadata(self, pages: List[Dict], year: int, lang: str) -> List[Dict]:
        all_text = []
        page_map = []
        
        for page in pages:
            text = page['text']
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            for para in paragraphs:
                all_text.append(para)
                page_map.append(page['page'])
        
        sections = self.section_extractor.extract_sections(all_text)
        
        chunks = []
        chunk_id = 0
        
        for section in sections:
            section_title = section['title']
            normalized_title = self.section_extractor.normalize_section_title(section_title)
            
            for para in section['paragraphs']:
                word_count = len(para.split())
                
                if word_count < self.min_words:
                    continue
                
                if word_count > self.max_words:
                    sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
                    current_chunk = []
                    current_words = 0
                    
                    for sent in sentences:
                        sent_words = len(sent.split())
                        if current_words + sent_words > self.max_words and current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            if len(chunk_text.split()) >= self.min_words:
                                chunks.append(self.create_chunk(
                                    chunk_id, chunk_text, year, lang, 
                                    section_title, normalized_title, 0
                                ))
                                chunk_id += 1
                            current_chunk = [sent]
                            current_words = sent_words
                        else:
                            current_chunk.append(sent)
                            current_words += sent_words
                    
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.split()) >= self.min_words:
                            chunks.append(self.create_chunk(
                                chunk_id, chunk_text, year, lang, 
                                section_title, normalized_title, 0
                            ))
                            chunk_id += 1
                else:
                    chunks.append(self.create_chunk(
                        chunk_id, para, year, lang, 
                        section_title, normalized_title, 0
                    ))
                    chunk_id += 1
        
        return chunks
    
    def create_chunk(self, chunk_id: int, text: str, year: int, lang: str,
                     section: str, normalized_section: str, page: int) -> Dict:
        return {
            'id': f"{lang}_{year}_para_{chunk_id:06d}",
            'text': text,
            'year': year,
            'lang': lang,
            'section': section,
            'section_normalized': normalized_section,
            'page': page,
            'word_count': len(text.split()),
            'char_count': len(text),
            'numbers': self.extract_numbers(text)
        }