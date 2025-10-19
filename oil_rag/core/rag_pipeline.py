import torch
from typing import Dict, List, Optional, Tuple
from oil_rag.retrieval.retriever import HybridRetriever
from oil_rag.models.dynamic_lora import DynamicLoRAModel
from oil_rag.models.dra_controller import DRAController, FeatureExtractor


class DRARAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: DynamicLoRAModel,
        dra_controller: DRAController,
        feature_extractor: FeatureExtractor,
        device: str = "cuda"
    ):
        self.retriever = retriever
        self.generator = generator
        self.dra_controller = dra_controller
        self.feature_extractor = feature_extractor
        self.device = device
        
        self.dra_controller.to(device)
    
    def generate_answer(
        self,
        query: str,
        max_length: int = 512,
        num_beams: int = 4,
        use_dra: bool = True,
        return_context: bool = False
    ) -> Dict:
        documents, retrieval_scores = self.retriever.retrieve(query)
        
        if not documents:
            return {
                'answer': 'No relevant documents found.',
                'rank': 0,
                'confidence': 0.0,
                'context': []
            }
        
        rerank_scores = None
        if self.retriever.reranker:
            rerank_scores = [doc.get('rerank_score', 0.0) for doc in documents]
        
        if use_dra:
            features = self.feature_extractor.extract_query_features(
                query,
                retrieval_scores,
                rerank_scores
            )
            
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            rank = self.dra_controller.predict_rank(features_tensor)
            self.generator.set_rank(rank)
        else:
            rank = self.generator.r_max
        
        context_text = self._format_context(documents)
        
        prompt = f"Question: {query}\n\nContext: {context_text}\n\nAnswer:"
        
        inputs = self.generator.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        answer = self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        confidence = self._compute_confidence(retrieval_scores, rerank_scores)
        
        result = {
            'answer': answer,
            'rank': rank,
            'confidence': confidence,
            'num_retrieved': len(documents)
        }
        
        if return_context:
            result['context'] = documents
        
        return result
    
    def _format_context(self, documents: List[Dict], max_docs: int = 5) -> str:
        context_parts = []
        for i, doc in enumerate(documents[:max_docs], 1):
            text = doc.get('text', '')
            context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)
    
    def _compute_confidence(
        self,
        retrieval_scores: List[float],
        rerank_scores: Optional[List[float]]
    ) -> float:
        if not retrieval_scores:
            return 0.0
        
        top_score = retrieval_scores[0]
        
        if rerank_scores and len(rerank_scores) > 0:
            top_score = max(top_score, max(rerank_scores))
        
        score_gap = 0.0
        if len(retrieval_scores) > 1:
            score_gap = retrieval_scores[0] - retrieval_scores[1]
        
        confidence = 0.7 * top_score + 0.3 * score_gap
        
        return min(confidence, 1.0)
    
    def batch_generate(
        self,
        queries: List[str],
        max_length: int = 512,
        use_dra: bool = True
    ) -> List[Dict]:
        results = []
        for query in queries:
            result = self.generate_answer(
                query,
                max_length=max_length,
                use_dra=use_dra
            )
            results.append(result)
        
        return results
    
    def get_rank_statistics(self, queries: List[str]) -> Dict:
        ranks = []
        
        for query in queries:
            documents, retrieval_scores = self.retriever.retrieve(query)
            
            if not documents:
                continue
            
            features = self.feature_extractor.extract_query_features(
                query,
                retrieval_scores
            )
            
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            rank = self.dra_controller.predict_rank(features_tensor)
            ranks.append(rank)
        
        if not ranks:
            return {}
        
        return {
            'mean_rank': sum(ranks) / len(ranks),
            'min_rank': min(ranks),
            'max_rank': max(ranks),
            'rank_distribution': {r: ranks.count(r) for r in set(ranks)}
        }


class StaticLoRABaseline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: DynamicLoRAModel,
        fixed_rank: int = 16,
        device: str = "cuda"
    ):
        self.retriever = retriever
        self.generator = generator
        self.fixed_rank = fixed_rank
        self.device = device
        
        self.generator.set_rank(fixed_rank)
    
    def generate_answer(
        self,
        query: str,
        max_length: int = 512,
        num_beams: int = 4,
        return_context: bool = False
    ) -> Dict:
        documents, retrieval_scores = self.retriever.retrieve(query)
        
        if not documents:
            return {
                'answer': 'No relevant documents found.',
                'rank': self.fixed_rank,
                'confidence': 0.0,
                'context': []
            }
        
        context_text = self._format_context(documents)
        prompt = f"Question: {query}\n\nContext: {context_text}\n\nAnswer:"
        
        inputs = self.generator.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        answer = self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result = {
            'answer': answer,
            'rank': self.fixed_rank,
            'confidence': retrieval_scores[0] if retrieval_scores else 0.0,
            'num_retrieved': len(documents)
        }
        
        if return_context:
            result['context'] = documents
        
        return result
    
    def _format_context(self, documents: List[Dict], max_docs: int = 5) -> str:
        context_parts = []
        for i, doc in enumerate(documents[:max_docs], 1):
            text = doc.get('text', '')
            context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)