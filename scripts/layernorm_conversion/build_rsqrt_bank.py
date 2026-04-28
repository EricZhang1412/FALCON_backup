"""Build per-LayerNorm rsqrt conversion configs for a HF model.

This script:
1) collects var+eps statistics per LayerNorm module on a text set;
2) writes one training YAML per LayerNorm (`configs/yaml/training/<config>.yaml`);
3) writes a manifest JSON used by training/validation scripts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-LN rsqrt conversion config bank.")
    p.add_argument("--hf-model", type=str, required=True)
    p.add_argument("--text-file", type=str, required=True)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--base-template", type=str, default="configs/yaml/training/rsqrt_conversion.yaml")
    p.add_argument("--config-prefix", type=str, default="rsqrt_ln")
    p.add_argument("--min-quantile", type=float, default=0.001)
    p.add_argument("--max-quantile", type=float, default=0.999)
    p.add_argument("--min-floor", type=float, default=1e-6)
    p.add_argument("--max-ceil", type=float, default=20000.0)
    p.add_argument("--manifest-out", type=str, default="outputs/ln_rsqrt_bank/manifest.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required. Install with `pip install transformers`.") from exc

    template_path = Path(args.base_template)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    training_dir = template_path.parent
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [ln.strip() for ln in Path(args.text_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    texts = lines[: args.max_samples]
    if not texts:
        raise ValueError("No valid lines in text file.")

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.float32).to(device)
    model.eval()

    enc = tok(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    ln_modules: list[tuple[str, torch.nn.LayerNorm]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            ln_modules.append((name, module))
    if not ln_modules:
        raise ValueError("No LayerNorm module found in model.")

    per_ln_vals: dict[str, list[torch.Tensor]] = {name: [] for name, _ in ln_modules}

    hooks = []
    for name, ln in ln_modules:
        eps = float(ln.eps)
        reduce_dims = tuple(range(-len(ln.normalized_shape), 0))

        def make_hook(nm: str, rdims, ep: float):
            def hook(_mod, inputs, _output):
                x = inputs[0].detach().float()
                mu = x.mean(dim=rdims, keepdim=True)
                var = ((x - mu) ** 2).mean(dim=rdims, keepdim=True)
                per_ln_vals[nm].append((var + ep).reshape(-1).cpu())
            return hook

        hooks.append(ln.register_forward_hook(make_hook(name, reduce_dims, eps)))

    with torch.no_grad():
        n = input_ids.size(0)
        for s in range(0, n, args.batch_size):
            e = min(n, s + args.batch_size)
            model(input_ids=input_ids[s:e], attention_mask=attention_mask[s:e])

    for h in hooks:
        h.remove()

    entries = []
    for ln_name, _ in ln_modules:
        vals = torch.cat(per_ln_vals[ln_name])
        q_lo = float(torch.quantile(vals, torch.tensor(args.min_quantile)))
        q_hi = float(torch.quantile(vals, torch.tensor(args.max_quantile)))
        x_min = max(args.min_floor, q_lo * 0.8)
        x_max = min(args.max_ceil, max(q_hi * 1.2, x_min * 10.0))

        cfg_name = f"{args.config_prefix}_{_safe_name(args.hf_model)}_{_safe_name(ln_name)}"
        cfg = json.loads(json.dumps(template))  # deep copy via json
        cfg["name"] = cfg_name
        cfg["target"]["name"] = "rsqrt"
        cfg["target"]["domain"]["x_min"] = float(x_min)
        cfg["target"]["domain"]["x_max"] = float(x_max)
        cfg["export"]["export_model_name"] = f"{cfg_name}_export"

        cfg_path = training_dir / f"{cfg_name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")

        entries.append(
            {
                "ln_name": ln_name,
                "config_name": cfg_name,
                "x_min": x_min,
                "x_max": x_max,
                "q_lo": q_lo,
                "q_hi": q_hi,
            }
        )

    manifest = {
        "hf_model": args.hf_model,
        "text_file": str(Path(args.text_file)),
        "base_template": str(template_path),
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(entries)} configs.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
