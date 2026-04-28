"""
Log-Timescale Initialization (LTI) for MBE neurons.

Instead of initializing lambda_v uniformly in raw parameter space,
LTI parameterizes thresholds via the characteristic timescale
gamma = -log(lambda_v), and spreads bases uniformly in log(gamma) space.
Threshold amplitudes alpha_v are set from input-distribution quantiles,
and readout weights are solved in closed form via ridge regression.

The best candidate is selected by exact initialized-state MSE screening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from falcon.conversion.config import ConversionTrainingConfig, ModelInitSpec
from falcon.conversion.neuron import TrainableMBENeuron


@dataclass
class LTICandidate:
    """One candidate threshold dictionary with its screening results."""
    gamma_min: float
    gamma_max: float
    log_alpha_v: torch.Tensor
    logit_lambda_v: torch.Tensor
    log_alpha_r: torch.Tensor
    logit_lambda_r: torch.Tensor
    log_alpha_d: torch.Tensor
    logit_lambda_d: torch.Tensor
    readout_weights: torch.Tensor
    initialized_mse: float
    mean_spikes: float
    dead_basis_count: int


@dataclass
class LTIResult:
    """Result of LTI search."""
    best_candidate: LTICandidate
    all_candidates: list[LTICandidate]
    best_index: int


def _simulate_basis_outputs(
    x: np.ndarray,
    T: int,
    alpha_v: np.ndarray,
    lambda_v: np.ndarray,
    alpha_r: np.ndarray,
    lambda_r: np.ndarray,
    alpha_d: np.ndarray,
    lambda_d: np.ndarray,
    polarities: list[str],
    fold_abs_input_for_polarities: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate MBE forward pass for all bases on a dense grid (vectorized over samples).

    Returns:
        basis_outputs: (M, K) array of per-basis accumulated outputs
        spike_counts: (M, K) array of spike counts per basis
        dead_mask: (K,) boolean array, True if basis never fires on any input
    """
    x_flat = x.reshape(-1)
    M = len(x_flat)
    K = len(alpha_v)
    basis_outputs = np.zeros((M, K), dtype=np.float64)
    spike_counts = np.zeros((M, K), dtype=np.float64)

    for n in range(K):
        pol = polarities[n]
        # Project input to polarity — vectorized over all samples
        if fold_abs_input_for_polarities:
            x_abs = np.abs(x_flat)
            if pol == "positive":
                membrane = x_abs.copy()
            else:
                membrane = -x_abs.copy()
        else:
            if pol == "positive":
                membrane = np.maximum(x_flat, 0.0).copy()
            else:
                membrane = np.minimum(x_flat, 0.0).copy()

        output_acc = np.zeros(M, dtype=np.float64)
        sc = np.zeros(M, dtype=np.float64)

        for t in range(T):
            thresh_mag = alpha_v[n] * (lambda_v[n] ** t)
            reset_mag = alpha_r[n] * (lambda_r[n] ** t)
            out_mag = alpha_d[n] * (lambda_d[n] ** t)

            if pol == "positive":
                fired = membrane > thresh_mag
            else:
                fired = membrane < -thresh_mag

            fired_f = fired.astype(np.float64)
            if pol == "positive":
                membrane -= reset_mag * fired_f
            else:
                membrane += reset_mag * fired_f
            output_acc += out_mag * fired_f
            sc += fired_f

        basis_outputs[:, n] = output_acc
        spike_counts[:, n] = sc

    dead_mask = spike_counts.sum(axis=0) == 0
    return basis_outputs, spike_counts, dead_mask


def _solve_ridge_readout(O: np.ndarray, y: np.ndarray, eta: float = 1e-6) -> np.ndarray:
    """Closed-form ridge regression: w* = (O^T O + eta I)^{-1} O^T y"""
    K = O.shape[1]
    OtO = O.T @ O + eta * np.eye(K)
    Oty = O.T @ y
    return np.linalg.solve(OtO, Oty)


def _generate_lti_schedules(
    gamma_min: float,
    gamma_max: float,
    num_pos: int,
    num_neg: int,
    x_pos: np.ndarray,
    x_neg_abs: np.ndarray,
    init: ModelInitSpec,
    fold_abs_input_for_polarities: bool = False,
) -> dict[str, Any]:
    """
    Generate threshold schedules from LTI parameters.

    Threshold lambda_v is derived from gamma = -log(lambda_v):
      - gamma values are uniform in log-space: log(gamma) in [log(gamma_min), log(gamma_max)]
      - lambda_v = exp(-gamma)

    Threshold alpha_v is set from input quantiles:
      - alpha_{v,k} = Quantile(|x_polarity|, (k+1)/(K+1))

    Reset and output schedules inherit from the YAML init spec ranges
    and are spread linearly across bases to preserve schedule diversity.
    """
    if fold_abs_input_for_polarities:
        num_total = num_pos + num_neg
        x_abs = x_pos if len(x_pos) > 0 else x_neg_abs
        if len(x_abs) > 0:
            x_floor = max(float(np.min(x_abs)), 1e-8)
            x_ceil = float(np.max(x_abs))
            alpha_v_all = np.exp(np.linspace(np.log(x_floor), np.log(x_ceil), num_total))
        else:
            alpha_v_all = np.linspace(0.1, 1.0, num_total)

        log_gamma_all = np.linspace(np.log(gamma_min), np.log(gamma_max), num_total)
        gamma_all = np.exp(log_gamma_all)
        lambda_v_all = np.exp(-gamma_all)
        lambda_v_all = np.clip(lambda_v_all, 1e-6, 1.0 - 1e-6)
        alpha_v_all = np.clip(alpha_v_all, 1e-6, None)

        alpha_v_pos = alpha_v_all[:num_pos]
        alpha_v_neg = alpha_v_all[num_pos:]
        lambda_v_pos = lambda_v_all[:num_pos]
        lambda_v_neg = lambda_v_all[num_pos:]
    else:
        # --- Positive half ---
        log_gamma = np.linspace(np.log(gamma_min), np.log(gamma_max), num_pos)
        gamma_pos = np.exp(log_gamma)
        lambda_v_pos = np.exp(-gamma_pos)

        q_ranks_pos = np.array([(k + 1) / (num_pos + 1) for k in range(num_pos)])
        if len(x_pos) > 0:
            # For one-sided positive inputs, linear quantiles under-cover the near-zero
            # high-curvature region (e.g., rsqrt). Use log-space anchors instead.
            if len(x_neg_abs) == 0:
                x_floor = max(float(np.min(x_pos)), 1e-8)
                x_ceil = float(np.max(x_pos))
                alpha_v_pos = np.exp(np.linspace(np.log(x_floor), np.log(x_ceil), num_pos))
            else:
                alpha_v_pos = np.quantile(x_pos, q_ranks_pos)
        else:
            alpha_v_pos = np.linspace(0.1, 1.0, num_pos)

        # Clamp: lambda_v must be in (0, 1)
        lambda_v_pos = np.clip(lambda_v_pos, 1e-6, 1.0 - 1e-6)
        alpha_v_pos = np.clip(alpha_v_pos, 1e-6, None)

        # --- Negative half ---
        log_gamma_neg = np.linspace(np.log(gamma_min), np.log(gamma_max), num_neg)
        gamma_neg = np.exp(log_gamma_neg)
        lambda_v_neg = np.exp(-gamma_neg)

        q_ranks_neg = np.array([(k + 1) / (num_neg + 1) for k in range(num_neg)])
        if len(x_neg_abs) > 0:
            if len(x_pos) == 0:
                x_floor = max(float(np.min(x_neg_abs)), 1e-8)
                x_ceil = float(np.max(x_neg_abs))
                alpha_v_neg = np.exp(np.linspace(np.log(x_floor), np.log(x_ceil), num_neg))
            else:
                alpha_v_neg = np.quantile(x_neg_abs, q_ranks_neg)
        else:
            alpha_v_neg = np.linspace(0.1, 1.0, num_neg)

        lambda_v_neg = np.clip(lambda_v_neg, 1e-6, 1.0 - 1e-6)
        alpha_v_neg = np.clip(alpha_v_neg, 1e-6, None)

    # --- Inherit reset/output from init spec ranges (basis-wise diverse) ---
    alpha_r_pos = np.linspace(init.positive.reset.alpha.min, init.positive.reset.alpha.max, num_pos)
    lambda_r_pos = np.linspace(init.positive.reset.lambda_.min, init.positive.reset.lambda_.max, num_pos)
    alpha_d_pos = np.linspace(init.positive.output.alpha.min, init.positive.output.alpha.max, num_pos)
    lambda_d_pos = np.linspace(init.positive.output.lambda_.min, init.positive.output.lambda_.max, num_pos)

    alpha_r_neg = np.linspace(init.negative.reset.alpha.min, init.negative.reset.alpha.max, num_neg)
    lambda_r_neg = np.linspace(init.negative.reset.lambda_.min, init.negative.reset.lambda_.max, num_neg)
    alpha_d_neg = np.linspace(init.negative.output.alpha.min, init.negative.output.alpha.max, num_neg)
    lambda_d_neg = np.linspace(init.negative.output.lambda_.min, init.negative.output.lambda_.max, num_neg)

    polarities = ["positive"] * num_pos + ["negative"] * num_neg

    return {
        "alpha_v": np.concatenate([alpha_v_pos, alpha_v_neg]),
        "lambda_v": np.concatenate([lambda_v_pos, lambda_v_neg]),
        "alpha_r": np.concatenate([alpha_r_pos, alpha_r_neg]),
        "lambda_r": np.concatenate([lambda_r_pos, lambda_r_neg]),
        "alpha_d": np.concatenate([alpha_d_pos, alpha_d_neg]),
        "lambda_d": np.concatenate([lambda_d_pos, lambda_d_neg]),
        "polarities": polarities,
    }


def run_lti_search(
    config: ConversionTrainingConfig,
    x: np.ndarray,
    y: np.ndarray,
    *,
    budget: int = 64,
    eta: float = 1e-6,
    gamma_range: tuple[float, float] = (0.01, 0.5),
    seed: int = 42,
) -> LTIResult:
    """
    Run Log-Timescale Initialization search.

    Args:
        config: training config with model/init specs
        x: dense input grid, shape (M, 1)
        y: dense target values, shape (M, 1)
        budget: number of candidate dictionaries to screen
        eta: ridge regularization
        gamma_range: (gamma_min_lo, gamma_max_hi) bounds for random search
        seed: random seed for reproducibility

    Returns:
        LTIResult with best candidate and all screening results
    """
    rng = np.random.RandomState(seed)
    T = config.model.T
    num_pos = config.model.num_basis // 2
    num_neg = config.model.num_basis // 2
    init = config.model.init

    # Subsample grid for faster screening (use at most 2048 points)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    max_screen_samples = 2048
    if len(x_flat) > max_screen_samples:
        idx = np.linspace(0, len(x_flat) - 1, max_screen_samples, dtype=int)
        x_screen = x_flat[idx].reshape(-1, 1)
        y_screen = y_flat[idx]
    else:
        x_screen = x.copy()
        y_screen = y_flat.copy()

    x_pos = x_flat[x_flat > 0]
    x_neg_abs = np.abs(x_flat[x_flat < 0])
    fold_abs = bool(config.model.fold_abs_input_for_polarities)
    if fold_abs:
        x_abs = np.abs(x_flat)
        x_pos = x_abs
        x_neg_abs = x_abs
    if len(x_pos) == 0 or len(x_neg_abs) == 0:
        missing = "negative" if len(x_neg_abs) == 0 else "positive"
        print(f"[LTI] Detected one-sided input support: missing {missing} polarity in training grid.")
        print("[LTI] Using log-space threshold anchors for the active polarity.")

    gmin_lo, gmax_hi = gamma_range
    candidates: list[LTICandidate] = []

    print(f"[LTI] Screening {budget} candidates (T={T}, K={config.model.num_basis})...")

    for b in range(budget):
        # Sample gamma bounds
        log_gmin = np.log(gmin_lo) + rng.rand() * (np.log(gmax_hi * 0.5) - np.log(gmin_lo))
        log_gmax = log_gmin + rng.rand() * (np.log(gmax_hi) - log_gmin)
        g_min = np.exp(log_gmin)
        g_max = np.exp(log_gmax)

        # Generate schedules
        schedules = _generate_lti_schedules(
            g_min,
            g_max,
            num_pos,
            num_neg,
            x_pos,
            x_neg_abs,
            init,
            fold_abs_input_for_polarities=fold_abs,
        )

        # Simulate forward pass on subsampled grid
        O, spike_counts, dead_mask = _simulate_basis_outputs(
            x_screen, T,
            schedules["alpha_v"], schedules["lambda_v"],
            schedules["alpha_r"], schedules["lambda_r"],
            schedules["alpha_d"], schedules["lambda_d"],
            schedules["polarities"],
            fold_abs_input_for_polarities=fold_abs,
        )

        # Solve ridge readout
        w = _solve_ridge_readout(O, y_screen, eta=eta)

        # Compute initialized MSE
        y_hat = O @ w
        mse = float(np.mean((y_hat - y_screen) ** 2))
        mean_spikes = float(spike_counts.mean())
        dead_count = int(dead_mask.sum())

        # Convert to torch tensors for loading into model
        log_alpha_v = torch.log(torch.tensor(schedules["alpha_v"], dtype=torch.float32))
        logit_lambda_v = torch.logit(torch.tensor(schedules["lambda_v"], dtype=torch.float32))
        log_alpha_r = torch.log(torch.tensor(schedules["alpha_r"], dtype=torch.float32))
        logit_lambda_r = torch.logit(torch.tensor(schedules["lambda_r"], dtype=torch.float32))
        log_alpha_d = torch.log(torch.tensor(schedules["alpha_d"], dtype=torch.float32))
        logit_lambda_d = torch.logit(torch.tensor(schedules["lambda_d"], dtype=torch.float32))
        readout_w = torch.tensor(w, dtype=torch.float32)

        candidates.append(LTICandidate(
            gamma_min=g_min, gamma_max=g_max,
            log_alpha_v=log_alpha_v, logit_lambda_v=logit_lambda_v,
            log_alpha_r=log_alpha_r, logit_lambda_r=logit_lambda_r,
            log_alpha_d=log_alpha_d, logit_lambda_d=logit_lambda_d,
            readout_weights=readout_w,
            initialized_mse=mse, mean_spikes=mean_spikes,
            dead_basis_count=dead_count,
        ))

    # Select best: lowest MSE, break ties by fewer spikes
    candidates.sort(key=lambda c: (c.initialized_mse, c.mean_spikes))
    best_idx = 0
    best = candidates[best_idx]

    print(f"[LTI] Best candidate: MSE={best.initialized_mse:.6f}, "
          f"mean_spikes={best.mean_spikes:.2f}, dead={best.dead_basis_count}, "
          f"gamma=[{best.gamma_min:.4f}, {best.gamma_max:.4f}]")

    return LTIResult(best_candidate=best, all_candidates=candidates, best_index=best_idx)


def apply_lti_to_model(model: TrainableMBENeuron, lti_result: LTIResult, init: ModelInitSpec) -> None:
    """
    Load LTI-selected schedules and readout weights into a TrainableMBENeuron.

    Only threshold schedules and readout weights are overwritten.
    Reset/output schedules keep their normal model initialization values.
    """
    best = lti_result.best_candidate

    with torch.no_grad():
        # Overwrite threshold-related schedule parameters
        model.log_alpha_v.copy_(best.log_alpha_v)
        model.log_lambda_v.copy_(best.logit_lambda_v)

        # Overwrite readout weights
        model.fc.weight.copy_(best.readout_weights.reshape(1, -1))

    print("[LTI] Applied to model: threshold schedules + readout initialized from best candidate.")


def configure_lti_finetune_trainability(
    model: TrainableMBENeuron,
    *,
    freeze_threshold: bool = False,
    freeze_readout: bool = False,
) -> None:
    """
    Configure trainability after LTI init:
      - optionally freeze threshold parameters
      - optionally freeze readout
      - train reset/output parameters
    """
    model.log_alpha_v.requires_grad_(not freeze_threshold)
    model.log_lambda_v.requires_grad_(not freeze_threshold)
    model.fc.weight.requires_grad_(not freeze_readout)

    model.log_alpha_r.requires_grad_(True)
    model.log_lambda_r.requires_grad_(True)
    model.log_alpha_d.requires_grad_(True)
    model.log_lambda_d.requires_grad_(True)

    print(
        "[LTI] Finetune params: "
        f"freeze_threshold={freeze_threshold}, freeze_readout={freeze_readout}, "
        "train reset/output=True"
    )


def configure_lti_finetune_from_result(model: TrainableMBENeuron, lti_result: LTIResult) -> None:
    """
    Adaptive trainability policy:
      - If dead bases exist after LTI search, unfreeze all parameters to allow recovery.
      - Otherwise use the default constrained finetune (freeze threshold + readout).
    """
    dead = int(lti_result.best_candidate.dead_basis_count)
    if dead > 0:
        print(f"[LTI] Detected dead bases ({dead}). Switch to full-parameter finetune for recovery.")
        configure_lti_finetune_trainability(
            model,
            freeze_threshold=False,
            freeze_readout=False,
        )
    else:
        configure_lti_finetune_trainability(
            model,
            freeze_threshold=False,
            freeze_readout=False,
        )
