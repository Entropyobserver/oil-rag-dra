from typing import Set, List, Dict
import json
from pathlib import Path


class TerminologyDictionary:
    def __init__(self, terminology_file: str = None):
        self.norwegian_terms = self._load_norwegian_terms()
        self.english_terms = self._load_english_terms()
        self.bilingual_map = self._create_bilingual_map()
        
        if terminology_file and Path(terminology_file).exists():
            self.load_from_file(terminology_file)
    
    def _load_norwegian_terms(self) -> Set[str]:
        return {
            'boring', 'brønn', 'brønnintegritet', 'oljeuthenting', 
            'formasjonstester', 'produksjonsrør', 'havbunn',
            'undervannsutstyr', 'plattform', 'flytende produksjon',
            'reservoar', 'hydrokarboner', 'feltet', 'lisens',
            'utvinning', 'seismikk', 'leting', 'prospekt',
            'subsea', 'offshore', 'petroleum', 'naturgass',
            'karbonfangst', 'elektrifisering', 'utslipp',
            'hse', 'sikkerhet', 'vedlikehold', 'drift',
            'investering', 'kostnader', 'inntekter', 'prosjekt',
            'bore', 'komplettering', 'intervensjon', 'plugging',
            'reservoarkarakterisering', 'reservoarsimulering',
            'produksjonsoptimalisering', 'trykkvedlikehold',
            'vanninjeksjon', 'gassinjeksjon', 'eor',
            'feltutvikling', 'konseptvalg', 'modning',
            'borerør', 'boreslam', 'sementering', 'casing',
            'blowout', 'preventer', 'stigerør', 'manifold'
        }
    
    def _load_english_terms(self) -> Set[str]:
        return {
            'drilling', 'well', 'wellbore', 'reservoir', 'hydrocarbon',
            'production', 'subsea', 'offshore', 'platform', 'rig',
            'exploration', 'seismic', 'prospect', 'field', 'license',
            'extraction', 'petroleum', 'crude', 'gas', 'oil',
            'injection', 'water', 'enhanced', 'recovery', 'eor',
            'completion', 'intervention', 'workover', 'plugging',
            'abandonment', 'decommissioning', 'facility', 'fpso',
            'pipeline', 'flowline', 'manifold', 'riser', 'umbilical',
            'blowout', 'preventer', 'bop', 'mud', 'cement',
            'casing', 'tubing', 'perforation', 'fracturing',
            'pressure', 'temperature', 'flow', 'rate', 'decline',
            'reserves', 'resources', 'recovery', 'factor',
            'simulation', 'modeling', 'characterization',
            'safety', 'environment', 'emission', 'carbon',
            'capture', 'storage', 'ccs', 'electrification'
        }
    
    def _create_bilingual_map(self) -> Dict[str, str]:
        return {
            'boring': 'drilling',
            'brønn': 'well',
            'brønnintegritet': 'well integrity',
            'havbunn': 'seabed',
            'undervannsutstyr': 'subsea equipment',
            'plattform': 'platform',
            'reservoar': 'reservoir',
            'hydrokarboner': 'hydrocarbons',
            'feltet': 'field',
            'utvinning': 'extraction',
            'leting': 'exploration',
            'naturgass': 'natural gas',
            'karbonfangst': 'carbon capture',
            'sikkerhet': 'safety',
            'vedlikehold': 'maintenance',
            'drift': 'operations',
            'investering': 'investment',
            'produksjonsoptimalisering': 'production optimization',
            'vanninjeksjon': 'water injection',
            'gassinjeksjon': 'gas injection',
            'feltutvikling': 'field development',
            'boreslam': 'drilling mud',
            'sementering': 'cementing',
            'stigerør': 'riser'
        }
    
    def is_domain_term(self, word: str, language: str = 'no') -> bool:
        word_lower = word.lower()
        
        if language == 'no':
            return word_lower in self.norwegian_terms
        elif language == 'en':
            return word_lower in self.english_terms
        else:
            return word_lower in self.norwegian_terms or word_lower in self.english_terms
    
    def get_translation(self, term: str, source_lang: str = 'no') -> str:
        term_lower = term.lower()
        
        if source_lang == 'no':
            return self.bilingual_map.get(term_lower, term)
        else:
            reverse_map = {v: k for k, v in self.bilingual_map.items()}
            return reverse_map.get(term_lower, term)
    
    def extract_terms(self, text: str, language: str = 'no') -> List[str]:
        words = text.lower().split()
        terms = []
        
        for word in words:
            clean_word = word.strip('.,!?;:()')
            if self.is_domain_term(clean_word, language):
                terms.append(clean_word)
        
        return terms
    
    def add_terms(self, terms: List[str], language: str = 'no'):
        if language == 'no':
            self.norwegian_terms.update(term.lower() for term in terms)
        elif language == 'en':
            self.english_terms.update(term.lower() for term in terms)
    
    def add_translation(self, norwegian_term: str, english_term: str):
        self.bilingual_map[norwegian_term.lower()] = english_term.lower()
    
    def save_to_file(self, filepath: str):
        data = {
            'norwegian_terms': list(self.norwegian_terms),
            'english_terms': list(self.english_terms),
            'bilingual_map': self.bilingual_map
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.norwegian_terms = set(data.get('norwegian_terms', []))
        self.english_terms = set(data.get('english_terms', []))
        self.bilingual_map = data.get('bilingual_map', {})
    
    def get_all_terms(self) -> Set[str]:
        return self.norwegian_terms.union(self.english_terms)
    
    def compute_term_density(self, text: str, language: str = 'no') -> float:
        words = text.split()
        if not words:
            return 0.0
        
        terms = self.extract_terms(text, language)
        return len(terms) / len(words)


def create_default_dictionary() -> TerminologyDictionary:
    return TerminologyDictionary()