"""One-click pipeline for per-LN rsqrt fitting and HF validation.

Pipeline:
1) scripts.layernorm_conversion.build_rsqrt_bank
2) scripts.layernorm_conversion.train_rsqrt_bank
3) scripts.model_validation.validate_mbe_inference (with per-LN rsqrt bank)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CONVERSION_CONFIG = "gelu_conversion"


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full per-LN rsqrt fitting + evaluation pipeline.")
    p.add_argument("--hf-model", type=str, default="gpt2")
    p.add_argument("--text-file", type=str, default="data/eval_texts_wikitext.txt")
    p.add_argument(
        "--wikitext-subset",
        type=str,
        default=None,
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="Use official WikiText split for eval (validate stage).",
    )
    p.add_argument(
        "--wikitext-split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Split for --wikitext-subset.",
    )
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])

    p.add_argument(
        "--comparison-root",
        type=str,
        default=f"outputs/comparison/{DEFAULT_CONVERSION_CONFIG}",
    )

    p.add_argument("--manifest-out", type=str, default="outputs/ln_rsqrt_bank/gpt2_manifest.json")
    p.add_argument("--config-prefix", type=str, default="rsqrt_ln")
    p.add_argument("--build-device", type=str, default="cpu")

    p.add_argument("--train-manual", action="store_true", help="Train only manual bank")
    p.add_argument("--train-lti", action="store_true", help="Train only lti bank")
    p.add_argument("--lti-budget", type=int, default=64)
    p.add_argument("--gamma-lo", type=float, default=0.01)
    p.add_argument("--gamma-hi", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0, help="Train first N LN configs (0 means all)")

    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--save-json", type=str, default="outputs/validation/ln_rsqrt_pipeline_report.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    py = sys.executable
    conversion_config = DEFAULT_CONVERSION_CONFIG

    # default: if neither flag is set, train both
    train_manual = args.train_manual or (not args.train_manual and not args.train_lti)
    train_lti = args.train_lti or (not args.train_manual and not args.train_lti)

    manifest_path = Path(args.manifest_out)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    if not args.skip_build:
        cmd = [
            py,
            "-m",
            "scripts.layernorm_conversion.build_rsqrt_bank",
            "--hf-model",
            args.hf_model,
            "--text-file",
            args.text_file,
            "--max-samples",
            str(args.max_samples),
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.build_device,
            "--config-prefix",
            args.config_prefix,
            "--manifest-out",
            str(manifest_path),
        ]
        _run(cmd, root)

    if not args.skip_train:
        cmd = [
            py,
            "-m",
            "scripts.layernorm_conversion.train_rsqrt_bank",
            "--manifest",
            str(manifest_path),
            "--budget",
            str(args.lti_budget),
            "--gamma-lo",
            str(args.gamma_lo),
            "--gamma-hi",
            str(args.gamma_hi),
            "--limit",
            str(args.limit),
        ]
        if train_manual:
            cmd.append("--manual")
        if train_lti:
            cmd.append("--lti")
        _run(cmd, root)

    if not args.skip_eval:
        cmd = [
            py,
            "-m",
            "scripts.model_validation.validate_mbe_inference",
            "--hf-model",
            args.hf_model,
            "--conversion-config",
            conversion_config,
            "--comparison-root",
            args.comparison_root,
            "--rsqrt-bank-manifest",
            str(manifest_path),
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--stride",
            str(args.stride),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--save-json",
            args.save_json,
        ]
        if args.wikitext_subset:
            cmd.extend(["--wikitext-subset", args.wikitext_subset, "--wikitext-split", args.wikitext_split])
            if args.max_samples > 0:
                cmd.extend(["--max-samples", str(args.max_samples)])
        else:
            cmd.extend(["--text-file", args.text_file, "--max-samples", str(args.max_samples)])
        _run(cmd, root)

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
