"""Apply LoRA to OpenCity encoder attention + FFN layers."""
import torch.nn as nn

from repro.lora.lora_linear import LoRALinear, count_lora_params

LORA_TARGET_NAMES = frozenset({
    "t_q_conv", "t_k_conv", "t_v_conv",
    "tc_q_conv", "tc_k_conv", "tc_v_conv",
    "w1", "w2", "w3",
})


def _replace_linear(parent: nn.Module, name: str, rank: int, alpha: float | None):
    old = getattr(parent, name)
    if not isinstance(old, nn.Linear):
        return
    setattr(parent, name, LoRALinear(old, rank=rank, alpha=alpha))


def apply_lora_to_opencity(predictor, rank: int, alpha: float | None = None) -> int:
    """Inject LoRA into encoder_blocks; return number of replaced layers."""
    n = 0
    if not hasattr(predictor, "encoder_blocks"):
        raise TypeError("expected OpenCity predictor")
    attn_names = ("t_q_conv", "t_k_conv", "t_v_conv", "tc_q_conv", "tc_k_conv", "tc_v_conv")
    ffn_names = ("w1", "w2", "w3")
    for block in predictor.encoder_blocks:
        st = block.st_attn
        for name in attn_names:
            if hasattr(st, name):
                _replace_linear(st, name, rank, alpha)
                n += 1
        mlp = block.mlp
        for name in ffn_names:
            if hasattr(mlp, name):
                _replace_linear(mlp, name, rank, alpha)
                n += 1
    return n


def freeze_non_lora(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True


def trainable_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def lora_param_count(module: nn.Module) -> int:
    return count_lora_params(module)
