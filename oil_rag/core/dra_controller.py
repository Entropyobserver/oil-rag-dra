from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalContext:
    query: str
    language: str
    retrieved_chunks: List[Dict]
    confidence: float
    iteration: int


class DRAController:
    def __init__(
        self,
        retriever,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        adaptive: bool = True
    ):
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.adaptive = adaptive
        self.history = []

    def should_continue_retrieval(self, context: RetrievalContext) -> bool:
        if context.iteration >= self.max_iterations:
            return False
        
        if context.confidence >= self.confidence_threshold:
            return False
        
        if not self.adaptive:
            return context.iteration < self.max_iterations
        
        return True

    def compute_confidence(self, chunks: List[Dict]) -> float:
        if not chunks:
            return 0.0
        
        scores = [c.get("score", 0.0) for c in chunks]
        avg_score = np.mean(scores)
        
        top_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
        consistency = np.std(scores[:3]) if len(scores) >= 3 else 1.0
        
        confidence = avg_score * 0.6 + top_gap * 0.3 + (1 - consistency) * 0.1
        return min(max(confidence, 0.0), 1.0)

    def retrieve(
        self,
        query: str,
        language: str = "en",
        top_k: int = 10
    ) -> RetrievalContext:
        context = None
        
        for iteration in range(self.max_iterations):
            chunks = self.retriever.retrieve(
                query=query,
                language=language,
                top_k=top_k
            )
            
            confidence = self.compute_confidence(chunks)
            
            context = RetrievalContext(
                query=query,
                language=language,
                retrieved_chunks=chunks,
                confidence=confidence,
                iteration=iteration
            )
            
            self.history.append(context)
            
            if not self.should_continue_retrieval(context):
                break
            
            query = self.refine_query(query, chunks)
        
        return context

    def refine_query(self, original_query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return original_query
        
        top_chunk = chunks[0]
        keywords = self.extract_keywords(top_chunk["text"])
        
        return f"{original_query} {' '.join(keywords[:3])}"

    def extract_keywords(self, text: str) -> List[str]:
        words = text.lower().split()
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        return [w for w in words if w not in stopwords and len(w) > 4][:5]
