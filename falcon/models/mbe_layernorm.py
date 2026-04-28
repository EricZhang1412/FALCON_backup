from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn


class ExactRSqrt(nn.Module):
    """Reference rsqrt module used as fallback."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x)


class MBERSqrtApproximator(nn.Module):
    """
    Wrap an MBE scalar approximator and expose it as a tensor-wise rsqrt module.

    The wrapped module is expected to map shape (N, 1) -> (N, 1).
    """

    def __init__(
        self,
        mbe_model: nn.Module,
        *,
        min_input: float = 1e-8,
        max_input: float | None = None,
    ) -> None:
        super().__init__()
        self.mbe_model = mbe_model
        self.min_input = float(min_input)
        self.max_input = None if max_input is None else float(max_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, min=self.min_input)
        if self.max_input is not None:
            x_safe = torch.clamp(x_safe, max=self.max_input)

        orig_shape = x_safe.shape
        flat = x_safe.reshape(-1, 1).to(dtype=torch.float32)
        y = self.mbe_model(flat)
        return y.reshape(orig_shape).to(dtype=x.dtype)


class MBERSqrtLayerNorm(nn.Module):
    """
    LayerNorm variant that keeps mean/variance exact but replaces rsqrt with a module.

    y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        rsqrt_module: nn.Module | None = None,
        min_var_eps_input: float = 1e-8,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(v) for v in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.min_var_eps_input = float(min_var_eps_input)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.rsqrt_module = rsqrt_module if rsqrt_module is not None else ExactRSqrt()

    @classmethod
    def from_layernorm(
        cls,
        layernorm: nn.LayerNorm,
        *,
        rsqrt_module: nn.Module,
        min_var_eps_input: float = 1e-8,
    ) -> "MBERSqrtLayerNorm":
        out = cls(
            normalized_shape=tuple(layernorm.normalized_shape),
            eps=float(layernorm.eps),
            elementwise_affine=bool(layernorm.elementwise_affine),
            bias=(layernorm.bias is not None),
            rsqrt_module=rsqrt_module,
            min_var_eps_input=min_var_eps_input,
        )
        if layernorm.elementwise_affine:
            with torch.no_grad():
                out.weight.copy_(layernorm.weight)
                if layernorm.bias is not None and out.bias is not None:
                    out.bias.copy_(layernorm.bias)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=reduce_dims, keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean(dim=reduce_dims, keepdim=True)

        var_eps = torch.clamp(var + self.eps, min=self.min_var_eps_input)
        inv_std = self.rsqrt_module(var_eps)
        y = centered * inv_std

        if self.elementwise_affine:
            y = y * self.weight
            if self.bias is not None:
                y = y + self.bias
        return y


def load_mbe_from_checkpoint(
    *,
    conversion_config_name: str,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    Build a TrainableMBENeuron and load checkpoint weights.

    This helper keeps import local to avoid hard dependency at module import time.
    """
    from falcon.conversion.config import load_conversion_training_config
    from falcon.conversion.neuron import TrainableMBENeuron

    cfg = load_conversion_training_config(conversion_config_name)
    dev = torch.device(device)
    model = TrainableMBENeuron(T=cfg.model.T, num_basis=cfg.model.num_basis, init=cfg.model.init).to(dev)
    payload = torch.load(Path(checkpoint_path), map_location=dev)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def replace_layernorm_with_mbe(
    module: nn.Module,
    *,
    rsqrt_module_factory: Callable[[], nn.Module],
) -> int:
    """Recursively replace nn.LayerNorm with MBERSqrtLayerNorm. Returns count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            new_ln = MBERSqrtLayerNorm.from_layernorm(child, rsqrt_module=rsqrt_module_factory())
            # Keep replacement module on the same device/dtype as the original LayerNorm.
            if child.elementwise_affine and child.weight is not None:
                new_ln = new_ln.to(device=child.weight.device, dtype=child.weight.dtype)
            else:
                # Fallback: infer from first available parameter/buffer in child.
                ref_tensor = next(child.parameters(), None)
                if ref_tensor is None:
                    ref_tensor = next(child.buffers(), None)
                if ref_tensor is not None:
                    new_ln = new_ln.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            setattr(module, name, new_ln)
            replaced += 1
        else:
            replaced += replace_layernorm_with_mbe(child, rsqrt_module_factory=rsqrt_module_factory)
    return replaced


def replace_layernorm_with_mbe_map(
    module: nn.Module,
    *,
    rsqrt_module_map: dict[str, nn.Module],
    _prefix: str = "",
) -> int:
    """
    Replace only selected LayerNorm modules by full module name.

    Args:
        module: root module
        rsqrt_module_map: mapping from full layernorm module name to rsqrt module
    """
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if isinstance(child, nn.LayerNorm):
            if full_name not in rsqrt_module_map:
                continue
            new_ln = MBERSqrtLayerNorm.from_layernorm(child, rsqrt_module=rsqrt_module_map[full_name])
            if child.elementwise_affine and child.weight is not None:
                new_ln = new_ln.to(device=child.weight.device, dtype=child.weight.dtype)
            else:
                ref_tensor = next(child.parameters(), None)
                if ref_tensor is None:
                    ref_tensor = next(child.buffers(), None)
                if ref_tensor is not None:
                    new_ln = new_ln.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            setattr(module, name, new_ln)
            replaced += 1
        else:
            replaced += replace_layernorm_with_mbe_map(
                child,
                rsqrt_module_map=rsqrt_module_map,
                _prefix=full_name,
            )
    return replaced
