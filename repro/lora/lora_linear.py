"""Minimal LoRA for nn.Linear (no peft dependency)."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + b + scaling * (B @ A @ x)"""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.linear = linear
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else 2 * rank)
        self.scaling = self.alpha / rank
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        dev = linear.weight.device
        dtype = linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out + lora


def count_lora_params(module: nn.Module) -> int:
    n = 0
    for m in module.modules():
        if isinstance(m, LoRALinear):
            n += m.lora_A.numel() + m.lora_B.numel()
    return n
