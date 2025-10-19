#!/usr/bin/env python3
"""
QA Dataset Generator for Oil Company Reports
Generates question-answer pairs from processed oil company documents.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oil_rag.utils.logger import setup_logger


class QAGenerator:
    """Generate QA pairs from oil company report documents."""
    
    def __init__(self, data_dir="data/processed/aligned"):
        self.logger = setup_logger('QAGenerator')
        self.data_dir = Path(data_dir)
        self.documents = []
        self.qa_pairs = []
        
        # Question templates by category
        self.question_templates = {
            'production': [
                "What was the oil production volume in {year}?",
                "How much gas was produced in {year}?",
                "What were the production figures for {year}?",
                "What was the daily production rate in {year}?",
                "How did production volumes change in {year}?",
            ],
            'financial': [
                "What was the revenue in {year}?",
                "What were the financial results for {year}?",
                "How did the company perform financially in {year}?",
                "What was the profit margin in {year}?",
                "What were the key financial metrics for {year}?",
            ],
            'safety': [
                "What safety measures were implemented in {year}?",
                "What were the safety statistics for {year}?",
                "How did the company ensure offshore safety in {year}?",
                "What safety improvements were made in {year}?",
                "What was the safety performance in {year}?",
            ],
            'environment': [
                "What environmental initiatives were taken in {year}?",
                "How did the company address climate change in {year}?",
                "What were the environmental impacts in {year}?",
                "What sustainability measures were implemented in {year}?",
                "What was the carbon footprint in {year}?",
            ],
            'technology': [
                "What new technologies were developed in {year}?",
                "How was technology used to improve operations in {year}?",
                "What research and development activities occurred in {year}?",
                "What technological innovations were introduced in {year}?",
                "How did digital transformation progress in {year}?",
            ],
            'operations': [
                "What were the main operational activities in {year}?",
                "How did operations change in {year}?",
                "What challenges were faced in operations during {year}?",
                "What operational improvements were made in {year}?",
                "How were offshore operations managed in {year}?",
            ],
            'strategy': [
                "What was the company strategy in {year}?",
                "How did the business strategy evolve in {year}?",
                "What were the strategic priorities for {year}?",
                "What strategic changes were announced in {year}?",
                "How did the company position itself in {year}?",
            ],
            'projects': [
                "What major projects were undertaken in {year}?",
                "What new developments started in {year}?",
                "What project milestones were achieved in {year}?",
                "What capital projects were approved in {year}?",
                "How did project execution perform in {year}?",
            ]
        }
        
        # Keywords for answer extraction
        self.answer_keywords = {
            'production': ['production', 'barrel', 'bcm', 'volume', 'output', 'boe'],
            'financial': ['revenue', 'profit', 'income', 'earnings', 'EBITDA', 'billion', 'million'],
            'safety': ['safety', 'incident', 'accident', 'HSE', 'security', 'risk'],
            'environment': ['environment', 'emission', 'carbon', 'climate', 'sustainability', 'CO2'],
            'technology': ['technology', 'digital', 'innovation', 'R&D', 'development', 'research'],
            'operations': ['operations', 'drilling', 'processing', 'facility', 'platform', 'field'],
            'strategy': ['strategy', 'strategic', 'vision', 'goal', 'objective', 'priority'],
            'projects': ['project', 'development', 'construction', 'investment', 'capex', 'facility']
        }
    
    def load_documents(self):
        """Load all aligned documents from the data directory."""
        self.logger.info(f"Loading documents from {self.data_dir}")
        
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    
                    # Extract English documents
                    if 'en' in doc and 'text' in doc['en']:
                        en_doc = doc['en'].copy()
                        en_doc['pair_id'] = doc['pair_id']
                        self.documents.append(en_doc)
        
        self.logger.info(f"Loaded {len(self.documents)} documents")
    
    def extract_year_from_doc(self, doc: Dict) -> int:
        """Extract year from document."""
        return doc.get('year', 2020)
    
    def find_relevant_sentences(self, text: str, keywords: List[str], max_sentences: int = 3) -> List[str]:
        """Find sentences containing relevant keywords."""
        sentences = re.split(r'[.!?]+', text)
        relevant = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains any keywords
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant.append(sentence)
                
                if len(relevant) >= max_sentences:
                    break
        
        return relevant
    
    def generate_answer_from_sentences(self, sentences: List[str]) -> str:
        """Generate a coherent answer from relevant sentences."""
        if not sentences:
            return "Information not available in the document."
        
        # Combine sentences and clean up
        answer = '. '.join(sentences)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Limit answer length
        if len(answer) > 300:
            answer = answer[:300] + "..."
        
        return answer
    
    def generate_qa_for_category(self, category: str, target_count: int = 25) -> List[Dict]:
        """Generate QA pairs for a specific category."""
        self.logger.info(f"Generating {target_count} QA pairs for category: {category}")
        
        qa_pairs = []
        templates = self.question_templates[category]
        keywords = self.answer_keywords[category]
        
        # Group documents by year
        docs_by_year = {}
        for doc in self.documents:
            year = self.extract_year_from_doc(doc)
            if year not in docs_by_year:
                docs_by_year[year] = []
            docs_by_year[year].append(doc)
        
        generated = 0
        attempts = 0
        max_attempts = target_count * 3
        
        while generated < target_count and attempts < max_attempts:
            attempts += 1
            
            # Select random year and document
            year = random.choice(list(docs_by_year.keys()))
            doc = random.choice(docs_by_year[year])
            
            # Generate question
            template = random.choice(templates)
            question = template.format(year=year)
            
            # Find relevant sentences in document
            relevant_sentences = self.find_relevant_sentences(
                doc['text'], keywords, max_sentences=2
            )
            
            if not relevant_sentences:
                continue
            
            # Generate answer
            answer = self.generate_answer_from_sentences(relevant_sentences)
            
            if len(answer) < 50:  # Skip very short answers
                continue
            
            qa_pair = {
                'id': f"{category}_{generated:03d}",
                'question': question,
                'answer': answer,
                'category': category,
                'year': year,
                'doc_id': doc['id'],
                'context': doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text']
            }
            
            qa_pairs.append(qa_pair)
            generated += 1
        
        self.logger.info(f"Generated {generated} QA pairs for {category}")
        return qa_pairs
    
    def generate_all_qa_pairs(self, total_target: int = 200) -> List[Dict]:
        """Generate QA pairs across all categories."""
        self.logger.info(f"Generating {total_target} total QA pairs")
        
        categories = list(self.question_templates.keys())
        pairs_per_category = total_target // len(categories)
        
        all_qa_pairs = []
        
        for category in categories:
            category_pairs = self.generate_qa_for_category(category, pairs_per_category)
            all_qa_pairs.extend(category_pairs)
        
        # Add extra pairs if needed to reach target
        remaining = total_target - len(all_qa_pairs)
        if remaining > 0:
            extra_category = random.choice(categories)
            extra_pairs = self.generate_qa_for_category(extra_category, remaining)
            all_qa_pairs.extend(extra_pairs)
        
        # Shuffle the final list
        random.shuffle(all_qa_pairs)
        
        self.logger.info(f"Generated {len(all_qa_pairs)} total QA pairs")
        return all_qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict], output_path: str = "data/test/test_qa.jsonl"):
        """Save QA pairs to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
    
    def generate_summary_report(self, qa_pairs: List[Dict]) -> Dict:
        """Generate a summary report of the QA dataset."""
        categories = {}
        years = {}
        
        for qa in qa_pairs:
            category = qa['category']
            year = qa['year']
            
            categories[category] = categories.get(category, 0) + 1
            years[year] = years.get(year, 0) + 1
        
        report = {
            'total_questions': len(qa_pairs),
            'categories': categories,
            'years': years,
            'avg_question_length': sum(len(qa['question']) for qa in qa_pairs) / len(qa_pairs),
            'avg_answer_length': sum(len(qa['answer']) for qa in qa_pairs) / len(qa_pairs)
        }
        
        return report


def main():
    """Main function to generate QA dataset."""
    print("Oil Company Reports QA Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = QAGenerator()
    
    # Load documents
    generator.load_documents()
    
    if not generator.documents:
        print("No documents found! Please check data directory.")
        return
    
    # Generate QA pairs
    qa_pairs = generator.generate_all_qa_pairs(total_target=200)
    
    # Save to file
    generator.save_qa_pairs(qa_pairs)
    
    # Generate and display report
    report = generator.generate_summary_report(qa_pairs)
    
    print("\nDataset Generation Complete!")
    print(f"Total Questions: {report['total_questions']}")
    print(f"Average Question Length: {report['avg_question_length']:.1f} chars")
    print(f"Average Answer Length: {report['avg_answer_length']:.1f} chars")
    
    print("\nQuestions by Category:")
    for category, count in report['categories'].items():
        print(f"  {category}: {count}")
    
    print("\nQuestions by Year:")
    for year, count in sorted(report['years'].items()):
        print(f"  {year}: {count}")
    
    print(f"\nDataset saved to: data/test/test_qa.jsonl")


if __name__ == "__main__":
    main()