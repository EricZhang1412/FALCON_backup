from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BasisSchedule:
    alpha_v: float
    lambda_v: float
    alpha_r: float
    lambda_r: float
    alpha_d: float
    lambda_d: float
    name: str = "basis"
    polarity: str = "positive"

    def __post_init__(self) -> None:
        positive = {"alpha_v": self.alpha_v, "alpha_r": self.alpha_r, "alpha_d": self.alpha_d}
        decays = {"lambda_v": self.lambda_v, "lambda_r": self.lambda_r, "lambda_d": self.lambda_d}
        for key, value in positive.items():
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")
        for key, value in decays.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{key} must lie in (0, 1), got {value}.")
        if self.polarity not in {"positive", "negative"}:
            raise ValueError(f"polarity must be 'positive' or 'negative', got {self.polarity}.")

    def project_input(self, x: float) -> float:
        return max(x, 0.0) if self.polarity == "positive" else min(x, 0.0)

    def threshold_at(self, t: int) -> float:
        magnitude = self.alpha_v * (self.lambda_v ** t)
        return magnitude if self.polarity == "positive" else -magnitude

    def reset_at(self, t: int) -> float:
        magnitude = self.alpha_r * (self.lambda_r ** t)
        return magnitude if self.polarity == "positive" else -magnitude

    def output_at(self, t: int) -> float:
        return self.alpha_d * (self.lambda_d ** t)

    def crosses_threshold(self, level: float, t: int) -> bool:
        threshold = self.threshold_at(t)
        return level > threshold if self.polarity == "positive" else level < threshold


@dataclass(frozen=True)
class MBENeuronConfig:
    basis_schedules: tuple[BasisSchedule, ...]
    basis_weights: tuple[float, ...]
    T: int
    dtype: torch.dtype = torch.float64
    device: str | torch.device = "cpu"
    verify_candidates: bool = True
    candidate_window: int = 2
    name: str = "mbe"

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if not self.basis_schedules:
            raise ValueError("At least one basis schedule is required.")
        if len(self.basis_schedules) != len(self.basis_weights):
            raise ValueError("basis_schedules and basis_weights must have the same length.")

    @property
    def num_bases(self) -> int:
        return len(self.basis_schedules)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def weight_tensor(self) -> torch.Tensor:
        return torch.tensor(self.basis_weights, dtype=self.dtype, device=self.torch_device)


SpikeTimeCollection = list[list[list[int]]]


@dataclass
class MBEResult:
    output: torch.Tensor
    spike_count: torch.Tensor | None = None
    spike_times: SpikeTimeCollection | None = None
    debug_traces: dict[str, Any] | None = None
