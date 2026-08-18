"""
LoRA layer implementation, plus injection into a real HuggingFace model.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a frozen base + a trainable low-rank
    adapter:

        output = base_linear(x) + scaling * (x @ A @ B)

    A is (in_features, r), B is (r, out_features). scaling is alpha/r or
    alpha/sqrt(r) depending on scaling_mode.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        scaling_mode: str = "alpha_over_r",
    ):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if scaling_mode == "alpha_over_r":
            self.scaling = alpha / r
        elif scaling_mode == "alpha_over_sqrt_r":
            self.scaling = alpha / math.sqrt(r)
        else:
            raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return base_out + self.scaling * lora_out

    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight; returns a plain nn.Linear."""
        with torch.no_grad():
            delta = self.scaling * (self.lora_A @ self.lora_B)  # (in, out)
            self.base.weight += delta.T  # nn.Linear.weight is (out, in)
        return self.base


def inject_lora(
    model: nn.Module,
    target_modules=("query", "value"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    scaling_mode: str = "alpha_over_r",
) -> nn.Module:
    """Recursively replace matching nn.Linear submodules with LoRALinear, in place."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(
                model,
                name,
                LoRALinear(module, r=r, alpha=alpha, dropout=dropout, scaling_mode=scaling_mode),
            )
        else:
            inject_lora(module, target_modules, r, alpha, dropout, scaling_mode)
    return model
