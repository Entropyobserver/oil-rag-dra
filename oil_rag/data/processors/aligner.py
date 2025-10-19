import numpy as np
from typing import List, Dict, Tuple, Set
from sentence_transformers import SentenceTransformer, util
import torch


class BilingualAligner:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        threshold: float = 0.70,
        section_threshold: float = 0.60,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        length_ratio_min: float = 0.5,
        length_ratio_max: float = 2.0,
        number_overlap_min: float = 0.5
    ):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.threshold = threshold
        self.section_threshold = section_threshold
        self.device = device
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
        self.number_overlap_min = number_overlap_min

    def encode_chunks(self, chunks: List[Dict]) -> np.ndarray:
        if not chunks:
            return np.array([])
        
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def match_sections(
        self,
        en_chunks: List[Dict],
        no_chunks: List[Dict]
    ) -> Dict[str, str]:
        en_sections = list(set(c["section_normalized"] for c in en_chunks))
        no_sections = list(set(c["section_normalized"] for c in no_chunks))
        
        if not en_sections or not no_sections:
            return {}
        
        en_section_embs = self.model.encode(en_sections, convert_to_numpy=True)
        no_section_embs = self.model.encode(no_sections, convert_to_numpy=True)
        
        similarity_matrix = util.cos_sim(en_section_embs, no_section_embs).numpy()
        
        section_map = {}
        for i, en_sec in enumerate(en_sections):
            best_idx = np.argmax(similarity_matrix[i])
            if similarity_matrix[i, best_idx] > self.section_threshold:
                section_map[en_sec] = no_sections[best_idx]
        
        return section_map

    def filter_by_section(
        self,
        en_chunk: Dict,
        no_chunks: List[Dict],
        section_map: Dict[str, str]
    ) -> List[int]:
        en_section = en_chunk["section_normalized"]
        
        if en_section not in section_map:
            return list(range(len(no_chunks)))
        
        target_section = section_map[en_section]
        indices = [
            i for i, c in enumerate(no_chunks)
            if c["section_normalized"] == target_section
        ]
        
        return indices if indices else list(range(len(no_chunks)))

    def check_length_ratio(self, en_chunk: Dict, no_chunk: Dict) -> bool:
        en_words = en_chunk.get("word_count", 0)
        no_words = no_chunk.get("word_count", 0)
        
        if en_words == 0 or no_words == 0:
            return False
        
        ratio = en_words / no_words
        return self.length_ratio_min <= ratio <= self.length_ratio_max

    def check_numbers_match(self, en_chunk: Dict, no_chunk: Dict) -> bool:
        en_numbers = set(en_chunk.get("numbers", []))
        no_numbers = set(no_chunk.get("numbers", []))
        
        if not en_numbers or not no_numbers:
            return True
        
        common = en_numbers.intersection(no_numbers)
        min_numbers = min(len(en_numbers), len(no_numbers))
        return len(common) >= min_numbers * self.number_overlap_min

    def check_bidirectional(
        self,
        en_idx: int,
        no_idx: int,
        en_embeddings: np.ndarray,
        no_embeddings: np.ndarray
    ) -> bool:
        no_emb = no_embeddings[no_idx:no_idx+1]
        similarities = util.cos_sim(no_emb, en_embeddings)[0].numpy()
        best_en_idx = np.argmax(similarities)
        return int(best_en_idx) == en_idx

    def compute_confidence(
        self,
        similarity: float,
        en_chunk: Dict,
        no_chunk: Dict,
        bidirectional: bool
    ) -> str:
        score = similarity
        
        if bidirectional:
            score += 0.05
        
        en_words = en_chunk.get("word_count", 0)
        no_words = no_chunk.get("word_count", 0)
        if no_words > 0:
            length_ratio = en_words / no_words
            if 0.7 <= length_ratio <= 1.3:
                score += 0.03
        
        if self.check_numbers_match(en_chunk, no_chunk):
            score += 0.02
        
        if score >= 0.85:
            return "high"
        elif score >= 0.75:
            return "medium"
        else:
            return "low"

    def align_chunks(
        self,
        en_chunks: List[Dict],
        no_chunks: List[Dict],
        en_embeddings: np.ndarray,
        no_embeddings: np.ndarray
    ) -> List[Dict]:
        if not en_chunks or not no_chunks:
            return []
        
        section_map = self.match_sections(en_chunks, no_chunks)
        alignments = []
        used_no_indices: Set[int] = set()
        
        for i, en_chunk in enumerate(en_chunks):
            candidate_indices = self.filter_by_section(
                en_chunk, no_chunks, section_map
            )
            candidate_indices = [
                idx for idx in candidate_indices
                if idx not in used_no_indices
            ]
            
            if not candidate_indices:
                continue
            
            en_emb = en_embeddings[i:i+1]
            candidate_embs = no_embeddings[candidate_indices]
            
            similarities = util.cos_sim(en_emb, candidate_embs)[0].numpy()
            best_local_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_local_idx])
            best_no_idx = candidate_indices[best_local_idx]
            
            if best_similarity < self.threshold:
                continue
            
            no_chunk = no_chunks[best_no_idx]
            
            if not self.check_length_ratio(en_chunk, no_chunk):
                continue
            
            if not self.check_numbers_match(en_chunk, no_chunk):
                continue
            
            bidirectional = self.check_bidirectional(
                i, best_no_idx, en_embeddings, no_embeddings
            )
            
            confidence = self.compute_confidence(
                best_similarity, en_chunk, no_chunk, bidirectional
            )
            
            year = en_chunk.get("year", 0)
            alignments.append({
                "pair_id": f"{year}_{i:06d}",
                "en": en_chunk,
                "no": no_chunk,
                "alignment": {
                    "score": best_similarity,
                    "method": "embedding_cosine",
                    "confidence": confidence,
                    "bidirectional_match": bidirectional
                }
            })
            
            used_no_indices.add(best_no_idx)
        
        return alignments