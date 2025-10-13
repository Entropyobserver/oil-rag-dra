import fitz
from pathlib import Path
from typing import List, Dict
import json


class PDFParser:
    def __init__(self):
        pass
    
    def parse_pdf(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({
                    'page': page_num + 1,
                    'text': text,
                    'source': Path(pdf_path).name
                })
        
        doc.close()
        return pages
    
    def parse_directory(self, dir_path: str) -> List[Dict]:
        pdf_files = Path(dir_path).glob('**/*.pdf')
        all_pages = []
        
        for pdf_file in pdf_files:
            try:
                pages = self.parse_pdf(str(pdf_file))
                all_pages.extend(pages)
            except Exception as e:
                print(f"Error parsing {pdf_file}: {e}")
        
        return all_pages
    
    def save_to_jsonl(self, pages: List[Dict], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in pages:
                f.write(json.dumps(page, ensure_ascii=False) + '\n')