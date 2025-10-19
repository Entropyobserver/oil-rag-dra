import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import re


class DRAController(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [32, 16],
        r_min: int = 4,
        r_max: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        alpha = self.network(features)
        
        rank_continuous = self.r_min + (self.r_max - self.r_min) * alpha
        
        rank_discrete = torch.round(rank_continuous / 4) * 4
        rank_discrete = torch.clamp(rank_discrete, self.r_min, self.r_max)
        
        return rank_discrete.int()
    
    def predict_rank(self, features: torch.Tensor) -> int:
        with torch.no_grad():
            rank = self.forward(features.unsqueeze(0))
            return int(rank.item())


class FeatureExtractor:
    def __init__(self, terminology_dict: List[str] = None):
        self.terminology_dict = set(terminology_dict) if terminology_dict else set()
    
    def extract_query_features(
        self,
        query: str,
        retrieval_scores: List[float],
        rerank_scores: List[float] = None
    ) -> np.ndarray:
        tokens = query.split()
        num_tokens = len(tokens)
        
        token_entropy = self._compute_token_entropy(tokens)
        
        term_density = self._compute_term_density(tokens)
        
        syntax_complexity = self._compute_syntax_complexity(query)
        
        top1_sim = retrieval_scores[0] if retrieval_scores else 0.0
        top10_var = np.var(retrieval_scores[:10]) if len(retrieval_scores) >= 10 else 0.0
        
        if rerank_scores:
            rerank_confidence = max(rerank_scores) if rerank_scores else 0.0
            retrieval_rerank_consistency = self._compute_overlap_score(
                retrieval_scores[:10],
                rerank_scores[:10]
            )
        else:
            rerank_confidence = top1_sim
            retrieval_rerank_consistency = 1.0
        
        features = np.array([
            self._normalize_token_count(num_tokens),
            term_density,
            syntax_complexity,
            token_entropy,
            top1_sim,
            top10_var,
            rerank_confidence,
            retrieval_rerank_consistency
        ], dtype=np.float32)
        
        return features
    
    def _normalize_token_count(self, count: int, max_count: int = 50) -> float:
        return min(count / max_count, 1.0)
    
    def _compute_token_entropy(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        total = len(tokens)
        entropy = 0.0
        
        for count in token_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        max_entropy = np.log2(len(tokens)) if len(tokens) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_term_density(self, tokens: List[str]) -> float:
        if not tokens or not self.terminology_dict:
            return 0.0
        
        term_count = sum(1 for token in tokens if token.lower() in self.terminology_dict)
        return term_count / len(tokens)
    
    def _compute_syntax_complexity(self, query: str) -> float:
        clauses = re.split(r'[,;]', query)
        num_clauses = len(clauses)
        
        words = query.split()
        avg_clause_length = len(words) / num_clauses if num_clauses > 0 else 0
        
        complexity = min((num_clauses - 1) * 0.3 + avg_clause_length / 20, 1.0)
        return complexity
    
    def _compute_overlap_score(
        self,
        scores1: List[float],
        scores2: List[float]
    ) -> float:
        if not scores1 or not scores2:
            return 0.0
        
        rank1 = np.argsort(np.argsort(scores1)[::-1])
        rank2 = np.argsort(np.argsort(scores2)[::-1])
        
        min_len = min(len(rank1), len(rank2))
        overlap = sum(1 for i in range(min_len) if rank1[i] == rank2[i])
        
        return overlap / min_len if min_len > 0 else 0.0


class DRAOptimizer:
    def __init__(
        self,
        controller: DRAController,
        lambda_dra: float = 0.1,
        beta_efficiency: float = 0.05
    ):
        self.controller = controller
        self.lambda_dra = lambda_dra
        self.beta_efficiency = beta_efficiency
    
    def compute_loss(
        self,
        generation_loss: torch.Tensor,
        predicted_rank: torch.Tensor,
        optimal_rank: torch.Tensor,
        quality_score: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        rank_loss = nn.MSELoss()(predicted_rank.float(), optimal_rank.float())
        
        efficiency_reward = (self.controller.r_max - predicted_rank.float()) * quality_score
        efficiency_reward = efficiency_reward.mean()
        
        dra_loss = rank_loss - self.beta_efficiency * efficiency_reward
        
        total_loss = generation_loss + self.lambda_dra * dra_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'generation_loss': generation_loss.item(),
            'rank_loss': rank_loss.item(),
            'efficiency_reward': efficiency_reward.item(),
            'dra_loss': dra_loss.item()
        }
        
        return total_loss, metrics