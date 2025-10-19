"""
Dynamic LoRA (Low-Rank Adaptation) Models

This module implements dynamic LoRA adaptations for different query complexities.
LoRA allows efficient fine-tuning by learning low-rank decomposition of weight updates.

The Dynamic LoRA system provides different parameter configurations optimized
for simple, medium, and complex queries in the oil & gas domain.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from .dra_controller import ComplexityLevel, DRAParameters
except ImportError:
    from dra_controller import ComplexityLevel, DRAParameters

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int                    # LoRA rank (bottleneck dimension)
    alpha: float                 # LoRA scaling factor
    dropout: float = 0.1         # Dropout rate
    target_modules: List[str] = None  # Target modules for adaptation
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for transformer models
            self.target_modules = ["query", "key", "value", "dense"]


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    
    Implements the LoRA technique: W = W_0 + BA
    where B ∈ R^{d×r}, A ∈ R^{r×k}, and r << min(d,k)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Track if layer is enabled
        self.enabled = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        if not self.enabled or self.rank == 0:
            return torch.zeros_like(x)
        
        # Apply LoRA: x @ A^T @ B^T * scaling
        result = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return self.dropout(result)
    
    def enable(self):
        """Enable LoRA adaptation."""
        self.enabled = True
    
    def disable(self):
        """Disable LoRA adaptation."""
        self.enabled = False
    
    def reset_parameters(self):
        """Reset LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)


class DynamicLoRAModule(nn.Module):
    """
    Dynamic LoRA Module that can switch between different LoRA configurations
    based on query complexity.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        lora_configs: Dict[ComplexityLevel, LoRAConfig]
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.lora_configs = lora_configs
        
        # Create LoRA layers for each complexity level
        self.lora_layers = nn.ModuleDict()
        
        for complexity, config in lora_configs.items():
            self.lora_layers[complexity.value] = LoRALayer(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout
            )
        
        # Current active complexity level
        self.current_complexity = ComplexityLevel.MEDIUM
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base layer and active LoRA layer."""
        # Base layer output
        base_output = self.base_layer(x)
        
        # Add LoRA adaptation
        lora_layer = self.lora_layers[self.current_complexity.value]
        lora_output = lora_layer(x)
        
        return base_output + lora_output
    
    def set_complexity(self, complexity: ComplexityLevel):
        """Set the current complexity level."""
        self.current_complexity = complexity
        logger.debug(f"LoRA complexity set to: {complexity.value}")
    
    def get_active_rank(self) -> int:
        """Get the rank of currently active LoRA layer."""
        return self.lora_layers[self.current_complexity.value].rank
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each complexity level."""
        counts = {}
        for complexity, lora_layer in self.lora_layers.items():
            rank = lora_layer.rank
            in_features = lora_layer.in_features
            out_features = lora_layer.out_features
            
            # LoRA parameters: A (rank × in_features) + B (out_features × rank)
            lora_params = rank * (in_features + out_features)
            counts[complexity] = lora_params
        
        return counts


class DynamicLoRAModel(nn.Module):
    """
    Complete Dynamic LoRA Model for RAG system.
    
    This model can dynamically adjust its capacity based on query complexity
    while maintaining efficiency through low-rank adaptations.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define LoRA configurations for each complexity level
        self.lora_configs = {
            ComplexityLevel.SIMPLE: LoRAConfig(rank=4, alpha=8.0, dropout=0.05),
            ComplexityLevel.MEDIUM: LoRAConfig(rank=8, alpha=16.0, dropout=0.1),
            ComplexityLevel.COMPLEX: LoRAConfig(rank=16, alpha=32.0, dropout=0.15)
        }
        
        # Build model layers with dynamic LoRA
        self.layers = nn.ModuleList()
        
        # Input projection
        input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.layers.append(
            DynamicLoRAModule(input_layer, self.lora_configs)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(
                DynamicLoRAModule(hidden_layer, self.lora_configs)
            )
        
        # Output projection
        output_layer = nn.Linear(hidden_dim, embedding_dim)
        self.layers.append(
            DynamicLoRAModule(output_layer, self.lora_configs)
        )
        
        # Layer normalization and activation
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else embedding_dim)
            for i in range(num_layers)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Current complexity level
        self.current_complexity = ComplexityLevel.MEDIUM
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dynamic LoRA model."""
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Apply layer with LoRA adaptation
            x = layer(x)
            
            # Apply normalization
            x = norm(x)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x
    
    def set_complexity(self, complexity: ComplexityLevel):
        """Set complexity level for all LoRA modules."""
        self.current_complexity = complexity
        
        for layer in self.layers:
            layer.set_complexity(complexity)
        
        logger.info(f"Model complexity set to: {complexity.value}")
    
    def adapt_from_parameters(self, params: DRAParameters):
        """Adapt model configuration from DRA parameters."""
        self.set_complexity(params.complexity_level)
        
        # Adjust dropout based on complexity
        if params.complexity_level == ComplexityLevel.SIMPLE:
            dropout_rate = 0.05
        elif params.complexity_level == ComplexityLevel.MEDIUM:
            dropout_rate = 0.1
        else:
            dropout_rate = 0.15
        
        # Update dropout layers
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get LoRA parameter counts for each complexity
        lora_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            layer_counts = layer.get_parameter_count()
            for complexity, count in layer_counts.items():
                if complexity not in lora_stats:
                    lora_stats[complexity] = 0
                lora_stats[complexity] += count
        
        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'current_complexity': self.current_complexity.value,
            'lora_parameters_by_complexity': lora_stats,
            'active_lora_rank': self.layers[0].get_active_rank(),
            'num_layers': self.num_layers,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim
        }
        
        return stats
    
    def save_lora_weights(self, path: str, complexity: ComplexityLevel):
        """Save LoRA weights for specific complexity level."""
        save_dict = {}
        
        for layer_idx, layer in enumerate(self.layers):
            lora_layer = layer.lora_layers[complexity.value]
            save_dict[f'layer_{layer_idx}_lora_A'] = lora_layer.lora_A.data
            save_dict[f'layer_{layer_idx}_lora_B'] = lora_layer.lora_B.data
        
        torch.save(save_dict, path)
        logger.info(f"LoRA weights saved for {complexity.value} to {path}")
    
    def load_lora_weights(self, path: str, complexity: ComplexityLevel):
        """Load LoRA weights for specific complexity level."""
        save_dict = torch.load(path, map_location='cpu')
        
        for layer_idx, layer in enumerate(self.layers):
            lora_layer = layer.lora_layers[complexity.value]
            
            lora_A_key = f'layer_{layer_idx}_lora_A'
            lora_B_key = f'layer_{layer_idx}_lora_B'
            
            if lora_A_key in save_dict:
                lora_layer.lora_A.data = save_dict[lora_A_key]
            if lora_B_key in save_dict:
                lora_layer.lora_B.data = save_dict[lora_B_key]
        
        logger.info(f"LoRA weights loaded for {complexity.value} from {path}")


class LoRATrainer:
    """
    Trainer for Dynamic LoRA models with complexity-specific optimization.
    """
    
    def __init__(
        self,
        model: DynamicLoRAModel,
        learning_rates: Dict[ComplexityLevel, float] = None
    ):
        self.model = model
        
        # Default learning rates for different complexity levels
        if learning_rates is None:
            learning_rates = {
                ComplexityLevel.SIMPLE: 1e-4,
                ComplexityLevel.MEDIUM: 5e-5,
                ComplexityLevel.COMPLEX: 1e-5
            }
        
        self.learning_rates = learning_rates
        
        # Create optimizers for each complexity level
        self.optimizers = {}
        for complexity in ComplexityLevel:
            # Only optimize LoRA parameters
            lora_params = []
            for layer in model.layers:
                lora_layer = layer.lora_layers[complexity.value]
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            
            self.optimizers[complexity] = torch.optim.AdamW(
                lora_params,
                lr=learning_rates[complexity],
                weight_decay=0.01
            )
    
    def train_complexity_level(
        self,
        complexity: ComplexityLevel,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        num_epochs: int = 5
    ):
        """Train LoRA parameters for specific complexity level."""
        
        self.model.set_complexity(complexity)
        optimizer = self.optimizers[complexity]
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_input, batch_target in train_data:
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch_input)
                
                # Compute loss (example: MSE)
                loss = nn.MSELoss()(output, batch_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            logger.info(f"Complexity {complexity.value}, Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")


if __name__ == "__main__":
    # Example usage and testing
    
    # Create dynamic LoRA model
    model = DynamicLoRAModel(
        embedding_dim=768,
        hidden_dim=512,
        num_layers=3
    )
    
    print("Dynamic LoRA Model Created")
    print("=" * 40)
    
    # Test different complexity levels
    test_input = torch.randn(2, 768)  # Batch of 2 embeddings
    
    for complexity in ComplexityLevel:
        model.set_complexity(complexity)
        
        output = model(test_input)
        stats = model.get_model_stats()
        
        print(f"\nComplexity: {complexity.value}")
        print(f"Active LoRA Rank: {stats['active_lora_rank']}")
        print(f"LoRA Parameters: {stats['lora_parameters_by_complexity'][complexity.value]}")
        print(f"Output Shape: {output.shape}")
    
    print(f"\nModel Statistics:")
    final_stats = model.get_model_stats()
    for key, value in final_stats.items():
        print(f"{key}: {value}")