from __future__ import annotations

from typing import Any

import torch

from falcon.models.interfaces import MBEResult, SpikeTimeCollection


def prepare_forward_inputs(x, *, dtype: torch.dtype, device: torch.device):
    x_tensor = torch.as_tensor(x, dtype=dtype, device=device)
    original_shape = x_tensor.shape
    flat_inputs = x_tensor.reshape(-1)
    return x_tensor, original_shape, flat_inputs, flat_inputs.numel(), x_tensor.numel()


def allocate_forward_buffers(*, num_samples, num_bases, dtype, device):
    basis_outputs = torch.zeros((num_samples, num_bases), dtype=dtype, device=device)
    spike_counts = torch.zeros((num_samples, num_bases), dtype=torch.int64, device=device)
    spike_times: SpikeTimeCollection = [[[] for _ in range(num_bases)] for _ in range(num_samples)]
    return basis_outputs, spike_counts, spike_times


def reshape_counts(counts, original_shape):
    return counts.reshape(counts.shape[-1]) if len(original_shape) == 0 else counts.reshape(*original_shape, counts.shape[-1])


def reshape_basis_outputs(basis_outputs, original_shape, num_bases):
    return basis_outputs.reshape(num_bases) if len(original_shape) == 0 else basis_outputs.reshape(*original_shape, num_bases)


def build_result(*, basis_outputs, spike_counts, spike_times, weight_tensor, original_shape, debug_traces=None):
    output = torch.matmul(basis_outputs, weight_tensor).reshape(original_shape)
    return MBEResult(
        output=output,
        spike_count=reshape_counts(spike_counts, original_shape),
        spike_times=spike_times,
        debug_traces=debug_traces,
    )
