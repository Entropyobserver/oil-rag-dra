from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


class AlignmentMetrics:
    @staticmethod
    def compute_statistics(
        alignments: List[Dict],
        en_chunks: List[Dict],
        no_chunks: List[Dict]
    ) -> Dict:
        total_en = len(en_chunks)
        total_no = len(no_chunks)
        total_aligned = len(alignments)
        
        if not alignments:
            return {
                "total_en_chunks": total_en,
                "total_no_chunks": total_no,
                "total_alignments": 0,
                "coverage_en": 0.0,
                "coverage_no": 0.0,
                "avg_similarity": 0.0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "bidirectional_matches": 0,
                "bidirectional_ratio": 0.0
            }
        
        scores = [a["alignment"]["score"] for a in alignments]
        avg_score = float(np.mean(scores))
        
        confidence_counts = defaultdict(int)
        for a in alignments:
            confidence_counts[a["alignment"]["confidence"]] += 1
        
        bidirectional = sum(
            1 for a in alignments
            if a["alignment"]["bidirectional_match"]
        )
        
        return {
            "total_en_chunks": total_en,
            "total_no_chunks": total_no,
            "total_alignments": total_aligned,
            "coverage_en": total_aligned / total_en if total_en > 0 else 0.0,
            "coverage_no": total_aligned / total_no if total_no > 0 else 0.0,
            "avg_similarity": avg_score,
            "high_confidence": confidence_counts["high"],
            "medium_confidence": confidence_counts["medium"],
            "low_confidence": confidence_counts["low"],
            "bidirectional_matches": bidirectional,
            "bidirectional_ratio": bidirectional / total_aligned if total_aligned > 0 else 0.0
        }


class RetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        if not retrieved or k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        relevant_retrieved = sum(
            1 for doc in top_k
            if doc.get("id") in relevant_set
        )
        
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        if not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        relevant_retrieved = sum(
            1 for doc in top_k
            if doc.get("id") in relevant_set
        )
        
        return relevant_retrieved / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[Dict], relevant: List[str]) -> float:
        relevant_set = set(relevant)
        
        for rank, doc in enumerate(retrieved, 1):
            if doc.get("id") in relevant_set:
                return 1.0 / rank
        
        return 0.0