"""
Comparison experiment: Manual MBE initialization vs LTI (Log-Timescale Initialization).

Runs both training pipelines on the same target function and produces:
  1. Loss convergence curves (both overlaid)
  2. Final fit quality comparison (target vs prediction)
  3. Summary statistics table

Usage:
    python -m scripts.single_op_conversion.compare_initializers [config_name] [--epochs N] [--budget B]

Example:
    python -m scripts.single_op_conversion.compare_initializers gelu_conversion --epochs 300 --budget 64
"""

from __future__ import annotations

import sys
import dataclasses
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from falcon.conversion.config import load_conversion_training_config
from falcon.conversion.data import build_conversion_dataloader, build_dense_eval_grid
from falcon.conversion.neuron import TrainableMBENeuron
from falcon.conversion.lti import run_lti_search, apply_lti_to_model, configure_lti_finetune_from_result
from falcon.conversion.train import train_conversion_model, evaluate_dense_grid
from falcon.utils.seeds import set_deterministic_seed

import torch


def build_model(config):
    model = TrainableMBENeuron(
        T=config.model.T,
        num_basis=config.model.num_basis,
        init=config.model.init,
        fold_abs_input_for_polarities=config.model.fold_abs_input_for_polarities,
    )
    return model.to(device=config.trainer.device, dtype=torch.float32)


def run_manual_training(config):
    """Train with standard manual initialization."""
    print("\n" + "=" * 60)
    print("[Manual Init] Starting training...")
    print("=" * 60)

    set_deterministic_seed(config.trainer.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_conversion_dataloader(config)
    model = build_model(config)

    # Evaluate initial state
    init_eval = evaluate_dense_grid(model, config)
    print(f"[Manual Init] Initial MSE: {init_eval['mse']:.6f}")

    result = train_conversion_model(model, loader, config)
    result["init_mse"] = init_eval["mse"]

    print(f"[Manual Init] Final MSE: {result['metrics']['final_mse']:.6f}, "
          f"Final MAE: {result['metrics']['final_mae']:.6f}")
    return result


def run_lti_training(config, lti_budget=64, gamma_range=(0.01, 0.5)):
    """Train with LTI initialization."""
    print("\n" + "=" * 60)
    print("[LTI Init] Starting training...")
    print("=" * 60)

    set_deterministic_seed(config.trainer.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_conversion_dataloader(config)
    model = build_model(config)

    # Run LTI search
    x, y = build_dense_eval_grid(config)
    lti_result = run_lti_search(
        config, x, y,
        budget=lti_budget,
        gamma_range=gamma_range,
        seed=config.trainer.seed,
    )
    apply_lti_to_model(model, lti_result, config.model.init)
    configure_lti_finetune_from_result(model, lti_result)

    # Evaluate initial state (after LTI)
    init_eval = evaluate_dense_grid(model, config)
    print(f"[LTI Init] Initial MSE (after LTI): {init_eval['mse']:.6f}")

    result = train_conversion_model(model, loader, config)
    result["init_mse"] = init_eval["mse"]
    result["lti_result"] = lti_result

    print(f"[LTI Init] Final MSE: {result['metrics']['final_mse']:.6f}, "
          f"Final MAE: {result['metrics']['final_mae']:.6f}")
    return result


def plot_comparison(manual_result, lti_result, config, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Style setup ──
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    manual_color = "#2196F3"  # blue
    lti_color = "#FF5722"     # orange-red
    target_color = "#424242"  # dark gray

    # ═══════════════════════════════════════════════════
    # Figure 1: Loss convergence curves
    # ═══════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    manual_epochs = [h["epoch"] for h in manual_result["history"]]
    manual_losses = [h["loss"] for h in manual_result["history"]]
    lti_epochs = [h["epoch"] for h in lti_result["history"]]
    lti_losses = [h["loss"] for h in lti_result["history"]]

    # Linear scale
    ax1.plot(manual_epochs, manual_losses, color=manual_color, linewidth=1.5, label="Manual Init")
    ax1.plot(lti_epochs, lti_losses, color=lti_color, linewidth=1.5, label="LTI Init")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (MSE)")
    ax1.set_title("Convergence: Linear Scale")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(manual_epochs, manual_losses, color=manual_color, linewidth=1.5, label="Manual Init")
    ax2.semilogy(lti_epochs, lti_losses, color=lti_color, linewidth=1.5, label="LTI Init")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss (MSE, log)")
    ax2.set_title("Convergence: Log Scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Training Convergence — {config.target.name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ═══════════════════════════════════════════════════
    # Figure 2: Fit quality comparison
    # ═══════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x_vals = manual_result["eval"]["x"].reshape(-1)
    target = manual_result["eval"]["target"].reshape(-1)
    manual_pred = manual_result["eval"]["prediction"].reshape(-1)
    lti_pred = lti_result["eval"]["prediction"].reshape(-1)

    # Top-left: Manual fit
    axes[0, 0].plot(x_vals, target, color=target_color, linewidth=2, label="Target", alpha=0.8)
    axes[0, 0].plot(x_vals, manual_pred, color=manual_color, linewidth=1.5, linestyle="--", label="Manual Init")
    axes[0, 0].set_title("Manual Init: Final Fit")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: LTI fit
    axes[0, 1].plot(x_vals, target, color=target_color, linewidth=2, label="Target", alpha=0.8)
    axes[0, 1].plot(x_vals, lti_pred, color=lti_color, linewidth=1.5, linestyle="--", label="LTI Init")
    axes[0, 1].set_title("LTI Init: Final Fit")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Both fits overlaid
    axes[1, 0].plot(x_vals, target, color=target_color, linewidth=2, label="Target", alpha=0.8)
    axes[1, 0].plot(x_vals, manual_pred, color=manual_color, linewidth=1.5, linestyle="--", label="Manual Init", alpha=0.8)
    axes[1, 0].plot(x_vals, lti_pred, color=lti_color, linewidth=1.5, linestyle=":", label="LTI Init", alpha=0.8)
    axes[1, 0].set_title("Both Fits Overlaid")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Absolute error comparison
    manual_err = np.abs(manual_pred - target)
    lti_err = np.abs(lti_pred - target)
    axes[1, 1].semilogy(x_vals, manual_err + 1e-12, color=manual_color, linewidth=1, alpha=0.7, label="Manual Init")
    axes[1, 1].semilogy(x_vals, lti_err + 1e-12, color=lti_color, linewidth=1, alpha=0.7, label="LTI Init")
    axes[1, 1].set_title("Pointwise Absolute Error (log scale)")
    axes[1, 1].set_ylabel("|error|")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Fit Quality — {config.target.name.upper()} (T={config.model.T}, K={config.model.num_basis})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fit_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ═══════════════════════════════════════════════════
    # Figure 3: Early convergence zoom (first 20% of epochs)
    # ═══════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 5))
    zoom_end = max(1, len(manual_epochs) // 5)
    ax.semilogy(manual_epochs[:zoom_end], manual_losses[:zoom_end], color=manual_color, linewidth=2, label="Manual Init", marker="o", markersize=2)
    ax.semilogy(lti_epochs[:zoom_end], lti_losses[:zoom_end], color=lti_color, linewidth=2, label="LTI Init", marker="s", markersize=2)

    # Mark initial MSE
    ax.axhline(y=manual_result["init_mse"], color=manual_color, linestyle=":", alpha=0.5, label=f"Manual init MSE = {manual_result['init_mse']:.4f}")
    ax.axhline(y=lti_result["init_mse"], color=lti_color, linestyle=":", alpha=0.5, label=f"LTI init MSE = {lti_result['init_mse']:.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title(f"Early Convergence (first {zoom_end} epochs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "early_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ═══════════════════════════════════════════════════
    # Summary table (printed + saved)
    # ═══════════════════════════════════════════════════
    summary = (
        f"\n{'='*60}\n"
        f"COMPARISON SUMMARY: {config.target.name.upper()}\n"
        f"T={config.model.T}, num_basis={config.model.num_basis}, epochs={config.trainer.max_epochs}\n"
        f"{'='*60}\n"
        f"{'Metric':<25} {'Manual Init':>15} {'LTI Init':>15} {'Improvement':>15}\n"
        f"{'-'*70}\n"
    )

    m_init_mse = manual_result["init_mse"]
    l_init_mse = lti_result["init_mse"]
    m_final_mse = manual_result["metrics"]["final_mse"]
    l_final_mse = lti_result["metrics"]["final_mse"]
    m_final_mae = manual_result["metrics"]["final_mae"]
    l_final_mae = lti_result["metrics"]["final_mae"]
    m_best_loss = manual_result["metrics"]["best_loss"]
    l_best_loss = lti_result["metrics"]["best_loss"]

    def ratio_str(manual_val, lti_val):
        if lti_val > 0:
            return f"{manual_val / lti_val:.2f}x"
        return "N/A"

    summary += f"{'Init MSE':<25} {m_init_mse:>15.6f} {l_init_mse:>15.6f} {ratio_str(m_init_mse, l_init_mse):>15}\n"
    summary += f"{'Final MSE':<25} {m_final_mse:>15.6f} {l_final_mse:>15.6f} {ratio_str(m_final_mse, l_final_mse):>15}\n"
    summary += f"{'Final MAE':<25} {m_final_mae:>15.6f} {l_final_mae:>15.6f} {ratio_str(m_final_mae, l_final_mae):>15}\n"
    summary += f"{'Best Train Loss':<25} {m_best_loss:>15.6f} {l_best_loss:>15.6f} {ratio_str(m_best_loss, l_best_loss):>15}\n"

    # Convergence speed: epoch at which loss drops below a threshold
    threshold = m_final_mse * 2.0  # When does each method reach 2x the manual final MSE?
    m_converge_epoch = next((h["epoch"] for h in manual_result["history"] if h["loss"] <= threshold), config.trainer.max_epochs)
    l_converge_epoch = next((h["epoch"] for h in lti_result["history"] if h["loss"] <= threshold), config.trainer.max_epochs)
    summary += f"{'Convergence epoch*':<25} {m_converge_epoch:>15.0f} {l_converge_epoch:>15.0f} {m_converge_epoch / max(l_converge_epoch, 1):.2f}x faster\n"
    summary += f"\n* Epoch at which loss first drops below {threshold:.6f} (2x manual final MSE)\n"

    if "lti_result" in lti_result:
        best = lti_result["lti_result"].best_candidate
        summary += f"\nLTI search details:\n"
        summary += f"  Dead bases: {best.dead_basis_count}\n"
        summary += f"  Mean spikes: {best.mean_spikes:.2f}\n"
        summary += f"  Gamma range: [{best.gamma_min:.4f}, {best.gamma_max:.4f}]\n"

    summary += f"{'='*60}\n"
    print(summary)

    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"\nPlots saved to: {output_dir}")


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]

    config_name = "gelu_conversion"
    max_epochs = None
    lti_budget = 64
    gamma_lo, gamma_hi = 0.01, 0.5

    i = 0
    while i < len(args):
        if args[i] == "--epochs" and i + 1 < len(args):
            max_epochs = int(args[i + 1]); i += 2
        elif args[i] == "--budget" and i + 1 < len(args):
            lti_budget = int(args[i + 1]); i += 2
        elif args[i] == "--gamma-lo" and i + 1 < len(args):
            gamma_lo = float(args[i + 1]); i += 2
        elif args[i] == "--gamma-hi" and i + 1 < len(args):
            gamma_hi = float(args[i + 1]); i += 2
        elif not args[i].startswith("--"):
            config_name = args[i]; i += 1
        else:
            i += 1

    config = load_conversion_training_config(config_name)

    # Apply epoch override
    if max_epochs is not None:
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, max_epochs=max_epochs))

    output_dir = Path("outputs/comparison") / config_name

    # --- Run Manual ---
    manual_config = dataclasses.replace(
        config,
        name=f"{config.name}_manual",
        artifacts=dataclasses.replace(config.artifacts, root_dir=str(output_dir / "manual")),
    )
    manual_result = run_manual_training(manual_config)

    # --- Run LTI ---
    lti_config = dataclasses.replace(
        config,
        name=f"{config.name}_lti",
        artifacts=dataclasses.replace(config.artifacts, root_dir=str(output_dir / "lti")),
    )
    lti_training_result = run_lti_training(lti_config, lti_budget=lti_budget, gamma_range=(gamma_lo, gamma_hi))

    # --- Plot comparison ---
    plot_comparison(manual_result, lti_training_result, config, output_dir / "plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
