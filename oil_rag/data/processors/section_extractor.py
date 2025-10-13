import re
from typing import List, Dict, Tuple


class SectionExtractor:
    def __init__(self):
        self.section_patterns = [
            r'^[A-Z][A-Za-z\s]{3,50}$',
            r'^\d+\.\s+[A-Z][A-Za-z\s]{3,50}$',
            r'^[A-Z]{2,}[\s\-]+[A-Z]{2,}',
        ]
    
    def is_section_header(self, text: str) -> bool:
        text = text.strip()
        
        if len(text) > 100 or len(text) < 3:
            return False
        
        if text.isupper() and len(text.split()) <= 8:
            return True
        
        for pattern in self.section_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def extract_sections(self, paragraphs: List[str]) -> List[Dict]:
        sections = []
        current_section = "Introduction"
        section_paragraphs = []
        
        for para in paragraphs:
            if self.is_section_header(para):
                if section_paragraphs:
                    sections.append({
                        'title': current_section,
                        'paragraphs': section_paragraphs
                    })
                current_section = para.strip()
                section_paragraphs = []
            else:
                section_paragraphs.append(para)
        
        if section_paragraphs:
            sections.append({
                'title': current_section,
                'paragraphs': section_paragraphs
            })
        
        return sections
    
    def normalize_section_title(self, title: str) -> str:
        title = title.lower().strip()
        title = re.sub(r'\d+\.?\s*', '', title)
        title = re.sub(r'[^\w\s]', '', title)
        return title