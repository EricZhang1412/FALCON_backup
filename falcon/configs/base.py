from __future__ import annotations

import torch

FALLBACK_DTYPE = torch.float64
FALLBACK_DEVICE = "cpu"
FALLBACK_SEED = 42
FALLBACK_T = 32


def coerce_dtype(value, default=FALLBACK_DTYPE):
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        mapping = {
            "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
            "fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64,
        }
        normalized = value.removeprefix("torch.").lower()
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"Unsupported dtype value: {value!r}")
