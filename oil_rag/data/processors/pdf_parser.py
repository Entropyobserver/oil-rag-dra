import fitz
from pathlib import Path
from typing import List, Dict, Optional


class PDFParser:
    def __init__(self, min_text_length: int = 10):
        self.min_text_length = min_text_length

    def parse_pdf(self, pdf_path: str) -> List[Dict]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        try:
            doc = fitz.open(str(path))
            pages = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip() and len(text.strip()) >= self.min_text_length:
                    pages.append({
                        "page": page_num + 1,
                        "text": text,
                        "source": path.name
                    })
            
            doc.close()
            return pages
        
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF {pdf_path}: {str(e)}")

    def parse_directory(
        self,
        dir_path: str,
        recursive: bool = True
    ) -> Dict[str, List[Dict]]:
        directory = Path(dir_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))
        
        results = {}
        errors = []
        
        for pdf_file in pdf_files:
            try:
                pages = self.parse_pdf(str(pdf_file))
                results[str(pdf_file)] = pages
            except Exception as e:
                errors.append((str(pdf_file), str(e)))
        
        if errors:
            print(f"Failed to parse {len(errors)} files:")
            for file_path, error in errors:
                print(f"  {file_path}: {error}")
        
        return results