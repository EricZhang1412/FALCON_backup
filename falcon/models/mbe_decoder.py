from __future__ import annotations

from typing import Any

import torch

from falcon.models._shared import allocate_forward_buffers, build_result, prepare_forward_inputs
from falcon.models.decoder_math import decode_spike_time
from falcon.models.interfaces import MBENeuronConfig, MBEResult


class MBENeuronDecoder:
    def __init__(self, config: MBENeuronConfig) -> None:
        self.config = config

    def _decode_one_basis(self, x_value, basis_idx, *, include_traces=False):
        schedule = self.config.basis_schedules[basis_idx]
        level = schedule.project_input(x_value)
        spike_times: list[int] = []
        basis_output = 0.0
        t_prev = -1

        while True:
            _, _, spike_t = decode_spike_time(
                level=level, schedule=schedule, t_prev=t_prev, T=self.config.T,
                verify_candidates=self.config.verify_candidates,
                candidate_window=self.config.candidate_window,
            )
            if spike_t is None:
                break
            spike_times.append(spike_t)
            basis_output += schedule.output_at(spike_t)
            level -= schedule.reset_at(spike_t)
            t_prev = spike_t
        return basis_output, spike_times

    def forward(self, x, *, include_traces=False) -> MBEResult:
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
            for basis_idx in range(num_bases):
                bo, bst = self._decode_one_basis(x_value, basis_idx)
                basis_outputs[sample_idx, basis_idx] = bo
                spike_counts[sample_idx, basis_idx] = len(bst)
                spike_times[sample_idx][basis_idx] = bst

        return build_result(
            basis_outputs=basis_outputs, spike_counts=spike_counts,
            spike_times=spike_times, weight_tensor=self.config.weight_tensor(),
            original_shape=original_shape,
        )

    __call__ = forward
