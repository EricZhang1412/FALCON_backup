"""Analyze spike statistics for a trained MBE checkpoint.

Usage examples:
  python -m scripts.single_op_conversion.analyze_spikes tanh_conversion --comparison-mode lti
  python -m scripts.single_op_conversion.analyze_spikes tanh_conversion --use-lti
  python -m scripts.single_op_conversion.analyze_spikes tanh_conversion --checkpoint /path/to/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from falcon.conversion.config import load_conversion_training_config
from falcon.conversion.data import sample_inputs
from falcon.conversion.neuron import TrainableMBENeuron


def _resolve_checkpoint_path(args, config) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)

    if args.comparison_mode is not None:
        mode = args.comparison_mode
        return (
            Path("outputs/comparison")
            / config.name
            / mode
            / f"{config.name}_{mode}"
            / config.checkpoint.dir
            / "best.pt"
        )

    if args.use_lti:
        root_dir = f"{config.artifacts.root_dir}_lti"
        run_name = f"{config.name}_lti"
    else:
        root_dir = config.artifacts.root_dir
        run_name = config.name
    return Path(root_dir) / run_name / config.checkpoint.dir / "best.pt"


@torch.no_grad()
def _collect_spike_stats(model: TrainableMBENeuron, x: torch.Tensor) -> dict[str, Any]:
    x = x.to(dtype=torch.float32)
    num_samples = int(x.shape[0])
    T = int(model.T)
    K = int(model.num_basis)
    Kp = int(model.num_pos_basis)

    lambda_v = torch.sigmoid(model.log_lambda_v).reshape(1, -1)
    lambda_r = torch.sigmoid(model.log_lambda_r).reshape(1, -1)
    lambda_d = torch.sigmoid(model.log_lambda_d).reshape(1, -1)

    threshold = model._signed_threshold(torch.exp(model.log_alpha_v).reshape(1, -1))
    reset = model._signed_reset(torch.exp(model.log_alpha_r).reshape(1, -1))
    output = torch.exp(model.log_alpha_d).reshape(1, -1)

    membrane = model._signed_membrane(x)
    spike_counts = torch.zeros((num_samples, K), device=x.device, dtype=torch.float32)
    per_timestep_rate: list[float] = []

    for _ in range(T):
        v_scaled = model._signed_vscaled(membrane, threshold)
        spikes = (v_scaled > 0).to(dtype=torch.float32)
        spike_counts += spikes

        per_timestep_rate.append(float(spikes.mean().item()))

        membrane = membrane - spikes * reset
        threshold = model._signed_threshold(torch.abs(threshold) * lambda_v)
        reset = model._signed_reset(torch.abs(reset) * lambda_r)
        output = output * lambda_d

    per_basis_mean_spikes = spike_counts.mean(dim=0)
    per_basis_firing_rate = per_basis_mean_spikes / float(T)
    dead_basis_mask = spike_counts.sum(dim=0) == 0

    mean_spikes = float(spike_counts.mean().item())
    firing_rate = mean_spikes / float(T)

    return {
        "num_samples": num_samples,
        "T": T,
        "num_basis": K,
        "overall": {
            "mean_spikes": mean_spikes,
            "firing_rate": firing_rate,
            "dead_basis_count": int(dead_basis_mask.sum().item()),
        },
        "polarity": {
            "positive": {
                "mean_spikes": float(per_basis_mean_spikes[:Kp].mean().item()),
                "firing_rate": float(per_basis_firing_rate[:Kp].mean().item()),
            },
            "negative": {
                "mean_spikes": float(per_basis_mean_spikes[Kp:].mean().item()),
                "firing_rate": float(per_basis_firing_rate[Kp:].mean().item()),
            },
        },
        "per_timestep_firing_rate": per_timestep_rate,
        "per_basis": [
            {
                "index": i,
                "polarity": "positive" if i < Kp else "negative",
                "mean_spikes": float(per_basis_mean_spikes[i].item()),
                "firing_rate": float(per_basis_firing_rate[i].item()),
                "dead": bool(dead_basis_mask[i].item()),
            }
            for i in range(K)
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze firing rate of a trained MBE model.")
    parser.add_argument("config_name", help="Training config name, e.g. tanh_conversion")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (best.pt)")
    parser.add_argument(
        "--comparison-mode",
        choices=["manual", "lti"],
        default=None,
        help="Load checkpoint from outputs/comparison/<config>/<mode>/...",
    )
    parser.add_argument("--use-lti", action="store_true", help="Load from outputs/conversion_lti/<name>_lti")
    parser.add_argument(
        "--grid",
        choices=["train", "eval"],
        default="eval",
        help="Which input grid to use for statistics.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Override grid sample count")
    parser.add_argument("--device", type=str, default=None, help="Override runtime device (e.g., cpu)")
    parser.add_argument("--save-json", type=str, default=None, help="Optional output JSON path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_conversion_training_config(args.config_name)

    device = torch.device(args.device or config.trainer.device)
    ckpt_path = _resolve_checkpoint_path(args, config)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = TrainableMBENeuron(
        T=config.model.T,
        num_basis=config.model.num_basis,
        init=config.model.init,
        fold_abs_input_for_polarities=config.model.fold_abs_input_for_polarities,
    )
    model = model.to(device=device, dtype=torch.float32)

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    default_n = config.data.eval_num_samples if args.grid == "eval" else config.data.num_samples
    n = args.num_samples or default_n
    x_np = sample_inputs(config, num_samples=n)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

    stats = _collect_spike_stats(model, x)
    stats["meta"] = {
        "config_name": args.config_name,
        "checkpoint": str(ckpt_path),
        "grid": args.grid,
        "device": str(device),
    }

    print("=" * 72)
    print("MBE Spike Statistics")
    print("=" * 72)
    print(f"config:      {stats['meta']['config_name']}")
    print(f"checkpoint:  {stats['meta']['checkpoint']}")
    print(f"grid/device: {stats['meta']['grid']} / {stats['meta']['device']}")
    print(f"samples:     {stats['num_samples']}, T={stats['T']}, basis={stats['num_basis']}")
    print("-" * 72)
    print(f"overall mean spikes: {stats['overall']['mean_spikes']:.6f}")
    print(f"overall firing rate: {stats['overall']['firing_rate']:.6f}")
    print(f"dead basis count:    {stats['overall']['dead_basis_count']}")
    print(
        "positive firing rate: "
        f"{stats['polarity']['positive']['firing_rate']:.6f} | "
        "negative firing rate: "
        f"{stats['polarity']['negative']['firing_rate']:.6f}"
    )
    print("=" * 72)

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"saved json: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
