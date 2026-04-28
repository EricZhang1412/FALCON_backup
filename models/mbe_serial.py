from __future__ import annotations

from typing import Any

import torch

from models._shared import allocate_forward_buffers, build_result, prepare_forward_inputs
from models.interfaces import MBENeuronConfig, MBEResult


class MBENeuronSerial:
    def __init__(self, config: MBENeuronConfig) -> None:
        self.config = config

    def forward(self, x, *, include_traces: bool = False) -> MBEResult:
        _, original_shape, flat_inputs, num_samples, _ = prepare_forward_inputs(
            x, dtype=self.config.dtype, device=self.config.torch_device,
        )
        num_bases = self.config.num_bases
        basis_outputs, spike_counts, spike_times = allocate_forward_buffers(
            num_samples=num_samples, num_bases=num_bases,
            dtype=self.config.dtype, device=self.config.torch_device,
        )

        for sample_idx in range(num_samples):
            x_value = float(flat_inputs[sample_idx].item())
            for basis_idx, schedule in enumerate(self.config.basis_schedules):
                membrane = schedule.project_input(x_value)
                basis_output = 0.0
                for t in range(self.config.T):
                    if schedule.crosses_threshold(membrane, t):
                        membrane -= schedule.reset_at(t)
                        basis_output += schedule.output_at(t)
                        spike_counts[sample_idx, basis_idx] += 1
                        spike_times[sample_idx][basis_idx].append(t)
                basis_outputs[sample_idx, basis_idx] = basis_output

        return build_result(
            basis_outputs=basis_outputs, spike_counts=spike_counts,
            spike_times=spike_times, weight_tensor=self.config.weight_tensor(),
            original_shape=original_shape,
        )

    __call__ = forward
