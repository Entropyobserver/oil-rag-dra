import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class DynamicLoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r_max
        
        self.lora_A = nn.Parameter(torch.randn(r_max, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))
        
        self.dropout = nn.Dropout(lora_dropout)
        
        self.current_rank = r_max
    
    def set_rank(self, rank: int):
        self.current_rank = min(rank, self.r_max)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_rank == 0:
            return torch.zeros(
                x.shape[0], x.shape[1], self.out_features,
                device=x.device,
                dtype=x.dtype
            )
        
        A = self.lora_A[:self.current_rank, :]
        B = self.lora_B[:, :self.current_rank]
        
        x_dropped = self.dropout(x)
        
        result = (x_dropped @ A.T) @ B.T
        
        return result * self.scaling


class DynamicLoRAModel:
    def __init__(
        self,
        base_model_name: str = "google/mt5-base",
        r_max: int = 32,
        lora_alpha: int = 32,
        target_modules: list = None,
        lora_dropout: float = 0.05
    ):
        self.base_model_name = base_model_name
        self.r_max = r_max
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q", "v"]
        self.lora_dropout = lora_dropout
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        self.lora_layers = {}
        self._inject_dynamic_lora()
        
        self.current_rank = r_max
    
    def _inject_dynamic_lora(self):
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = DynamicLoRALayer(
                        module.in_features,
                        module.out_features,
                        self.r_max,
                        self.lora_alpha,
                        self.lora_dropout
                    )
                    self.lora_layers[name] = lora_layer
                    
                    original_forward = module.forward
                    
                    def make_forward(orig_forward, lora):
                        def forward(x):
                            base_output = orig_forward(x)
                            lora_output = lora(x)
                            return base_output + lora_output
                        return forward
                    
                    module.forward = make_forward(original_forward, lora_layer)
    
    def set_rank(self, rank: int):
        self.current_rank = min(rank, self.r_max)
        for lora_layer in self.lora_layers.values():
            lora_layer.set_rank(self.current_rank)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs
    ) -> torch.Tensor:
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
    
    def save_pretrained(self, save_directory: str):
        self.base_model.save_pretrained(save_directory)
        
        lora_state = {
            'lora_layers': {name: layer.state_dict() for name, layer in self.lora_layers.items()},
            'config': {
                'r_max': self.r_max,
                'lora_alpha': self.lora_alpha,
                'target_modules': self.target_modules,
                'lora_dropout': self.lora_dropout
            }
        }
        
        torch.save(lora_state, f"{save_directory}/dynamic_lora.pt")
    
    def load_pretrained(self, load_directory: str):
        lora_state = torch.load(f"{load_directory}/dynamic_lora.pt")
        
        for name, layer in self.lora_layers.items():
            if name in lora_state['lora_layers']:
                layer.load_state_dict(lora_state['lora_layers'][name])
    
    def count_trainable_parameters(self) -> int:
        total = 0
        for layer in self.lora_layers.values():
            total += layer.lora_A.numel() + layer.lora_B.numel()
        return total
    
    def get_memory_footprint(self) -> dict:
        base_memory = sum(p.numel() * p.element_size() for p in self.base_model.parameters())
        lora_memory = sum(
            layer.lora_A.numel() * layer.lora_A.element_size() +
            layer.lora_B.numel() * layer.lora_B.element_size()
            for layer in self.lora_layers.values()
        )
        
        return {
            'base_model_mb': base_memory / (1024 ** 2),
            'lora_params_mb': lora_memory / (1024 ** 2),
            'total_mb': (base_memory + lora_memory) / (1024 ** 2)
        }