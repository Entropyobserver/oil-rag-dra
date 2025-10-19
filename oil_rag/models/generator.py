import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from typing import List, Dict, Optional


class RAGGenerator:
    def __init__(self, model_name='google/flan-t5-large', 
                 device=None, max_length=512):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if 't5' in model_name.lower() or 'flan' in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_type = 'seq2seq'
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = 'causal'
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, query: str, contexts: List[Dict], 
                 temperature=0.7, top_p=0.9):
        prompt = self._build_prompt(query, contexts)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=2048,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if self.model_type == 'causal':
            answer = answer[len(prompt):].strip()
        
        return answer
    
    def _build_prompt(self, query: str, contexts: List[Dict]):
        context_str = ""
        for i, ctx in enumerate(contexts[:5], 1):
            meta = ctx['metadata']
            text = meta.get('text', '')
            year = meta.get('year', 'N/A')
            section = meta.get('section', 'N/A')
            
            context_str += f"[{i}] Year {year}, Section: {section}\n{text}\n\n"
        
        prompt = f"""Based on the following context from oil industry reports, answer the question accurately and concisely.

Context:
{context_str}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_with_citations(self, query: str, contexts: List[Dict]):
        answer = self.generate(query, contexts)
        citations = self._extract_citations(answer, contexts)
        
        return {
            'answer': answer,
            'citations': citations,
            'num_contexts': len(contexts)
        }
    
    def _extract_citations(self, answer: str, contexts: List[Dict]):
        citations = []
        
        for i, ctx in enumerate(contexts):
            meta = ctx['metadata']
            text = meta.get('text', '')
            
            words = text.lower().split()[:20]
            match_count = sum(1 for w in words if w in answer.lower())
            
            if match_count >= 3:
                citations.append({
                    'context_id': i,
                    'year': meta.get('year'),
                    'section': meta.get('section'),
                    'score': ctx.get('score', 0)
                })
        
        return citations


class ConfidenceEstimator:
    def __init__(self):
        self.uncertainty_phrases = [
            'i dont know', 'not sure', 'unclear', 'cannot determine',
            'no information', 'not mentioned', 'difficult to say'
        ]
    
    def estimate(self, answer: str, contexts: List[Dict], query: str):
        score = 0.0
        
        word_count = len(answer.split())
        if 30 <= word_count <= 200:
            score += 0.25
        elif word_count < 10:
            score -= 0.2
        
        if self._contains_numbers(answer):
            score += 0.25
        
        if self._contains_uncertainty(answer):
            score -= 0.3
        
        if len(contexts) >= 3:
            score += 0.2
        
        if contexts:
            avg_context_score = sum(c.get('score', 0) for c in contexts) / len(contexts)
            score += avg_context_score * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _contains_numbers(self, text: str):
        return any(char.isdigit() for char in text)
    
    def _contains_uncertainty(self, text: str):
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.uncertainty_phrases)
    
    def is_confident(self, answer: str, contexts: List[Dict], 
                    query: str, threshold=0.7):
        confidence = self.estimate(answer, contexts, query)
        return confidence >= threshold