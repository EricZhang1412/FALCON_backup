from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from conversion.config import ConversionTrainingConfig

TargetFn = Callable[[np.ndarray], np.ndarray]

TARGETS: dict[str, TargetFn] = {
    "relu": lambda x: np.maximum(x, 0.0),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))),
    "elu": lambda x: np.where(x > 0.0, x, np.exp(x) - 1.0),
    "expm1": lambda x: np.expm1(x),
    "tanh": lambda x: np.tanh(x),
    "rsqrt": lambda x: 1.0 / np.sqrt(np.clip(x, 1e-8, None)),
}


def get_target_function(name: str) -> TargetFn:
    if name not in TARGETS:
        raise ValueError(f"Unsupported target function: {name}")
    return TARGETS[name]


def evaluate_target(name: str, inputs: np.ndarray) -> np.ndarray:
    return get_target_function(name)(inputs).astype(np.float32, copy=False)


def sample_inputs(config: ConversionTrainingConfig, *, num_samples: int | None = None) -> np.ndarray:
    n = num_samples or config.data.num_samples
    return np.linspace(config.target.x_min, config.target.x_max, n, dtype=np.float32).reshape(-1, 1)


def build_conversion_dataloader(config: ConversionTrainingConfig) -> DataLoader:
    x = sample_inputs(config)
    y = evaluate_target(config.target.name, x)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=config.data.batch_size, shuffle=config.data.shuffle)


def build_dense_eval_grid(config: ConversionTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    x = sample_inputs(config, num_samples=config.data.eval_num_samples)
    y = evaluate_target(config.target.name, x)
    return x, y
