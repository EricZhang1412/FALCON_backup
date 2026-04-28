"""Validate HF causal LM before/after MBE activation replacement.

Compares three variants on the same text set:
  1) original HF model
  2) MBE-manual activation
  3) MBE-LTI activation

Metrics:
  - perplexity (standard WikiText protocol: concatenate + stride)
  - last-token logits cosine similarity vs original
  - last-token top1 agreement vs original

Example:
  python validate_hf_mbe_inference.py \
    --hf-model gpt2 \
    --conversion-config tanh_conversion \
    --comparison-root outputs/comparison/tanh_conversion \
    --text-file data/eval_texts.txt \
    --max-samples 64 --max-length 1024 --stride 512
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from conversion.config import load_conversion_training_config
from conversion.neuron import TrainableMBENeuron
from models.mbe_layernorm import replace_layernorm_with_mbe, replace_layernorm_with_mbe_map


ACT_ALIASES = {
    "relu": {"relu"},
    "gelu": {"gelu", "gelu_new"},
    "silu": {"silu", "swish"},
    "tanh": {"tanh"},
    "elu": {"elu"},
    "expm1": {"expm1"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF model validation for MBE replacements.")
    parser.add_argument("--hf-model", type=str, required=True, help="HF model id or local model path.")
    parser.add_argument("--conversion-config", type=str, required=True, help="conversion yaml name, e.g. tanh_conversion")
    parser.add_argument(
        "--comparison-root",
        type=str,
        required=True,
        help="comparison root, e.g. outputs/comparison/tanh_conversion",
    )
    parser.add_argument("--manual-checkpoint", type=str, default=None, help="override manual best.pt path")
    parser.add_argument("--lti-checkpoint", type=str, default=None, help="override lti best.pt path")
    parser.add_argument(
        "--rsqrt-conversion-config",
        type=str,
        default=None,
        help="conversion config name for LayerNorm rsqrt MBE checkpoints (set together with --rsqrt-comparison-root to enable)",
    )
    parser.add_argument(
        "--rsqrt-comparison-root",
        type=str,
        default=None,
        help="comparison root for rsqrt conversion checkpoints (set together with --rsqrt-conversion-config to enable)",
    )
    parser.add_argument("--rsqrt-manual-checkpoint", type=str, default=None, help="override rsqrt manual best.pt path")
    parser.add_argument("--rsqrt-lti-checkpoint", type=str, default=None, help="override rsqrt lti best.pt path")
    parser.add_argument(
        "--rsqrt-bank-manifest",
        type=str,
        default=None,
        help="JSON manifest for per-LN independent rsqrt fitting.",
    )
    parser.add_argument(
        "--rsqrt-bank-manual-root",
        type=str,
        default="outputs/conversion",
        help="Root directory of per-LN manual checkpoints.",
    )
    parser.add_argument(
        "--rsqrt-bank-lti-root",
        type=str,
        default="outputs/conversion_lti",
        help="Root directory of per-LN lti checkpoints.",
    )
    parser.add_argument("--text-file", type=str, default=None, help="newline-separated evaluation texts")
    parser.add_argument(
        "--wikitext-subset",
        type=str,
        default=None,
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="official WikiText subset to evaluate on (overrides --text-file)",
    )
    parser.add_argument(
        "--wikitext-split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="split for --wikitext-subset",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="maximum number of text samples (0 means all)")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--max-length", type=int, default=128, help="max token length")
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="stride for concatenate+sliding-window WikiText PPL evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="runtime device: auto|cpu|cuda|cuda:0 ... (auto will fallback to cpu on cuda init failure)",
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--save-json", type=str, default=None, help="optional output json path")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "auto":
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Trigger lazy init early so failures can be caught here.
                _ = torch.empty(1, device="cuda")
                return torch.device("cuda")
        except Exception as exc:
            print(f"[warn] CUDA unavailable, fallback to cpu: {exc}")
        return torch.device("cpu")

    if req.startswith("cuda"):
        try:
            _ = torch.empty(1, device=req)
            return torch.device(req)
        except Exception as exc:
            print(f"[warn] Requested device '{requested}' not usable, fallback to cpu: {exc}")
            return torch.device("cpu")

    return torch.device(req)


def _resolve_checkpoint_pair(
    root: str | Path,
    config_name: str,
    manual_override: str | None,
    lti_override: str | None,
) -> tuple[Path, Path]:
    root = Path(root)
    manual = Path(manual_override) if manual_override else (
        root / "manual" / f"{config_name}_manual" / "checkpoints" / "best.pt"
    )
    lti = Path(lti_override) if lti_override else (
        root / "lti" / f"{config_name}_lti" / "checkpoints" / "best.pt"
    )
    if not manual.exists():
        raise FileNotFoundError(f"manual checkpoint not found: {manual}")
    if not lti.exists():
        raise FileNotFoundError(f"lti checkpoint not found: {lti}")
    return manual, lti


def _load_eval_texts(path: Path, max_samples: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"no valid lines in text file: {path}")
    if max_samples > 0:
        return lines[:max_samples]
    return lines


def _load_wikitext_texts(subset: str, split: str, max_samples: int) -> list[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets is required for --wikitext-subset. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("wikitext", subset, split=split)
    texts = [str(item.get("text", "")) for item in ds]
    if max_samples > 0:
        texts = texts[:max_samples]
    if not texts:
        raise ValueError(f"empty WikiText split: subset={subset}, split={split}")
    return texts


def _build_concatenated_corpus(texts: list[str]) -> str:
    # Standard WikiText-style evaluation concatenates the whole corpus,
    # then applies a sliding window over tokenized ids.
    return "\n\n".join(texts)


class MBEActivation(torch.nn.Module):
    """Element-wise MBE activation wrapper for arbitrary tensor shapes."""

    def __init__(self, mbe_model: TrainableMBENeuron) -> None:
        super().__init__()
        self.mbe = mbe_model
        self.mbe.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        orig_shape = x.shape
        flat = x.reshape(-1, 1).to(dtype=torch.float32)
        with torch.no_grad():
            y = self.mbe(flat)
        return y.reshape(orig_shape).to(dtype=in_dtype)


class SharedMBERSqrt(torch.nn.Module):
    """Shared rsqrt approximator callable wrapper for LayerNorm replacement."""

    def __init__(
        self,
        mbe_forward,
        *,
        min_input: float = 1e-8,
        max_input: float | None = None,
    ) -> None:
        super().__init__()
        self._mbe_forward = mbe_forward
        self.min_input = float(min_input)
        self.max_input = None if max_input is None else float(max_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, min=self.min_input)
        if self.max_input is not None:
            x_safe = torch.clamp(x_safe, max=self.max_input)
        orig_shape = x_safe.shape
        flat = x_safe.reshape(-1, 1).to(dtype=torch.float32)
        with torch.no_grad():
            y = self._mbe_forward(flat)
        return y.reshape(orig_shape).to(dtype=x.dtype)


def _load_rsqrt_bank_modules(
    manifest_path: Path,
    *,
    variant: str,
    manual_root: Path,
    lti_root: Path,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(f"Invalid or empty rsqrt bank manifest: {manifest_path}")

    out: dict[str, torch.nn.Module] = {}
    for item in entries:
        ln_name = str(item["ln_name"])
        cfg_name = str(item["config_name"])
        if variant == "manual":
            ckpt = manual_root / cfg_name / "checkpoints" / "best.pt"
        else:
            ckpt = lti_root / f"{cfg_name}_lti" / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{variant} rsqrt-bank checkpoint missing for {ln_name}: {ckpt}")
        mbe = _load_mbe_from_checkpoint(ckpt, cfg_name, device)
        out[ln_name] = SharedMBERSqrt(mbe.forward, min_input=1e-8)
    return out


def _callable_name(obj) -> str:
    if hasattr(obj, "__name__"):
        return str(obj.__name__).lower()
    return obj.__class__.__name__.lower()


def _match_activation(current_act, target_name: str) -> bool:
    aliases = ACT_ALIASES.get(target_name.lower(), {target_name.lower()})
    name = _callable_name(current_act)
    return any(alias in name for alias in aliases)


def _patch_model_activations(model, target_name: str, replacement: torch.nn.Module) -> int:
    replaced = 0
    for module in model.modules():
        for attr in ("act", "act_fn", "activation_fn"):
            if not hasattr(module, attr):
                continue
            cur = getattr(module, attr)
            if _match_activation(cur, target_name):
                setattr(module, attr, replacement)
                replaced += 1
    return replaced


def _load_mbe_from_checkpoint(
    ckpt_path: Path,
    conversion_cfg_name: str,
    device: torch.device,
) -> TrainableMBENeuron:
    cfg = load_conversion_training_config(conversion_cfg_name)
    mbe = TrainableMBENeuron(T=cfg.model.T, num_basis=cfg.model.num_basis, init=cfg.model.init).to(device=device)
    payload = torch.load(ckpt_path, map_location=device)
    mbe.load_state_dict(payload["model_state_dict"])
    mbe.eval()
    return mbe


@dataclass
class EvalAccumulator:
    nll_sum: float = 0.0
    loss_token_count: int = 0
    cos_sum: float = 0.0
    cos_count: int = 0
    top1_match: int = 0
    top1_total: int = 0

    def update_loss_from_labels(self, loss: torch.Tensor, labels: torch.Tensor) -> None:
        # HF causal LM loss shifts labels internally by one position.
        # Effective loss token count is (#unmasked labels - batch_size).
        valid_tokens = int((labels != -100).sum().item())
        batch_size = int(labels.shape[0])
        loss_tokens = max(valid_tokens - batch_size, 0)
        self.nll_sum += float(loss.item()) * float(loss_tokens)
        self.loss_token_count += loss_tokens

    def update_logit_alignment(self, base_logits: torch.Tensor, var_logits: torch.Tensor, attention_mask: torch.Tensor) -> None:
        bsz = base_logits.shape[0]
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(bsz, 1, base_logits.shape[-1])
        base_last = base_logits.gather(1, gather_idx).squeeze(1)
        var_last = var_logits.gather(1, gather_idx).squeeze(1)

        cos = F.cosine_similarity(base_last, var_last, dim=-1)
        self.cos_sum += float(cos.sum().item())
        self.cos_count += int(cos.numel())

        base_top1 = base_last.argmax(dim=-1)
        var_top1 = var_last.argmax(dim=-1)
        self.top1_match += int((base_top1 == var_top1).sum().item())
        self.top1_total += int(base_top1.numel())

    def summarize(self) -> dict:
        mean_loss = self.nll_sum / max(self.loss_token_count, 1)
        return {
            "mean_loss": mean_loss,
            "perplexity": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
            "last_token_logits_cosine": self.cos_sum / max(self.cos_count, 1),
            "last_token_top1_agreement": self.top1_match / max(self.top1_total, 1),
            "loss_token_count": self.loss_token_count,
            "num_sequences": self.top1_total,
        }


def _iter_batches(encodings: dict[str, torch.Tensor], batch_size: int) -> Iterable[dict[str, torch.Tensor]]:
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    n = input_ids.shape[0]
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        yield {
            "input_ids": input_ids[s:e],
            "attention_mask": attention_mask[s:e],
        }


def _iter_stride_windows(
    input_ids: torch.Tensor,
    *,
    max_length: int,
    stride: int,
) -> Iterable[dict[str, torch.Tensor]]:
    """Yield sliding windows for standard WikiText PPL evaluation."""
    seq_len = int(input_ids.shape[1])
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        window_input_ids = input_ids[:, begin:end]
        labels = window_input_ids.clone()
        if trg_len < labels.shape[1]:
            labels[:, :-trg_len] = -100
        attention_mask = torch.ones_like(window_input_ids)

        yield {
            "input_ids": window_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        prev_end = end
        if end == seq_len:
            break


def main() -> int:
    args = _parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for this script. Install it first, e.g. `pip install transformers`."
        ) from exc

    device = _resolve_device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    run_dtype = dtype_map[args.dtype]
    if device.type == "cpu" and run_dtype != torch.float32:
        print(f"[warn] dtype={args.dtype} on cpu may be unsupported/slow; force float32.")
        run_dtype = torch.float32

    conv_cfg = load_conversion_training_config(args.conversion_config)
    target_act = conv_cfg.target.name.lower()
    manual_ckpt, lti_ckpt = _resolve_checkpoint_pair(
        args.comparison_root,
        conv_cfg.name,
        args.manual_checkpoint,
        args.lti_checkpoint,
    )
    if args.wikitext_subset:
        texts = _load_wikitext_texts(args.wikitext_subset, args.wikitext_split, args.max_samples)
        eval_source = f"wikitext/{args.wikitext_subset}:{args.wikitext_split}"
    else:
        if not args.text_file:
            raise ValueError("Either --text-file or --wikitext-subset must be provided.")
        texts = _load_eval_texts(Path(args.text_file), args.max_samples)
        eval_source = str(Path(args.text_file))
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.max_length <= 1:
        raise ValueError("--max-length must be > 1 for causal LM perplexity.")
    if args.stride > args.max_length:
        print(f"[warn] stride ({args.stride}) > max_length ({args.max_length}); using stride=max_length.")
        args.stride = args.max_length

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    corpus_text = _build_concatenated_corpus(texts)
    enc = tokenizer(corpus_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Load original model once, then deep-copy for manual/lti variants.
    base_model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=run_dtype).to(device)
    base_model.eval()
    manual_model = copy.deepcopy(base_model)
    lti_model = copy.deepcopy(base_model)

    manual_mbe = _load_mbe_from_checkpoint(manual_ckpt, args.conversion_config, device)
    lti_mbe = _load_mbe_from_checkpoint(lti_ckpt, args.conversion_config, device)
    manual_replaced = _patch_model_activations(manual_model, target_act, MBEActivation(manual_mbe))
    lti_replaced = _patch_model_activations(lti_model, target_act, MBEActivation(lti_mbe))
    manual_ln_replaced = 0
    lti_ln_replaced = 0

    rsqrt_cfg_name = args.rsqrt_conversion_config
    rsqrt_root = args.rsqrt_comparison_root
    rsqrt_bank_manifest = Path(args.rsqrt_bank_manifest) if args.rsqrt_bank_manifest else None

    if rsqrt_bank_manifest is not None and (rsqrt_cfg_name or rsqrt_root):
        raise ValueError(
            "Use either --rsqrt-bank-manifest (per-LN mode) OR (--rsqrt-conversion-config + --rsqrt-comparison-root), not both."
        )

    if rsqrt_bank_manifest is None and (bool(rsqrt_cfg_name) != bool(rsqrt_root)):
        raise ValueError(
            "LayerNorm rsqrt replacement requires BOTH --rsqrt-conversion-config and --rsqrt-comparison-root."
        )

    enable_layernorm_rsqrt = (rsqrt_bank_manifest is not None) or bool(rsqrt_cfg_name and rsqrt_root)
    layernorm_mode = "none"

    if rsqrt_bank_manifest is not None:
        layernorm_mode = "per_ln_bank"
        manual_map = _load_rsqrt_bank_modules(
            rsqrt_bank_manifest,
            variant="manual",
            manual_root=Path(args.rsqrt_bank_manual_root),
            lti_root=Path(args.rsqrt_bank_lti_root),
            device=device,
        )
        lti_map = _load_rsqrt_bank_modules(
            rsqrt_bank_manifest,
            variant="lti",
            manual_root=Path(args.rsqrt_bank_manual_root),
            lti_root=Path(args.rsqrt_bank_lti_root),
            device=device,
        )
        manual_ln_replaced = replace_layernorm_with_mbe_map(manual_model, rsqrt_module_map=manual_map)
        lti_ln_replaced = replace_layernorm_with_mbe_map(lti_model, rsqrt_module_map=lti_map)
        print(f"[LayerNorm/per-LN] manual replaced={manual_ln_replaced}, lti replaced={lti_ln_replaced}")
    elif enable_layernorm_rsqrt:
        layernorm_mode = "shared_rsqrt"
        rsqrt_cfg = load_conversion_training_config(rsqrt_cfg_name)
        rsqrt_manual_ckpt, rsqrt_lti_ckpt = _resolve_checkpoint_pair(
            rsqrt_root,
            rsqrt_cfg.name,
            args.rsqrt_manual_checkpoint,
            args.rsqrt_lti_checkpoint,
        )
        rsqrt_manual_mbe = _load_mbe_from_checkpoint(rsqrt_manual_ckpt, rsqrt_cfg_name, device)
        rsqrt_lti_mbe = _load_mbe_from_checkpoint(rsqrt_lti_ckpt, rsqrt_cfg_name, device)

        manual_ln_replaced = replace_layernorm_with_mbe(
            manual_model,
            rsqrt_module_factory=lambda: SharedMBERSqrt(rsqrt_manual_mbe.forward, min_input=1e-8),
        )
        lti_ln_replaced = replace_layernorm_with_mbe(
            lti_model,
            rsqrt_module_factory=lambda: SharedMBERSqrt(rsqrt_lti_mbe.forward, min_input=1e-8),
        )
        print(f"[LayerNorm] manual replaced={manual_ln_replaced}, lti replaced={lti_ln_replaced}")

    if manual_replaced == 0 or lti_replaced == 0:
        raise RuntimeError(
            f"No activation module matched target '{target_act}'. "
            "Consider extending _patch_model_activations for this HF architecture."
        )

    base_acc = EvalAccumulator()
    manual_acc = EvalAccumulator()
    lti_acc = EvalAccumulator()

    with torch.no_grad():
        for batch in _iter_stride_windows(
            input_ids,
            max_length=args.max_length,
            stride=args.stride,
        ):
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            labels = batch["labels"]

            base_out = base_model(**model_inputs, labels=labels)
            man_out = manual_model(**model_inputs, labels=labels)
            lti_out = lti_model(**model_inputs, labels=labels)

            base_acc.update_loss_from_labels(base_out.loss, labels)
            manual_acc.update_loss_from_labels(man_out.loss, labels)
            lti_acc.update_loss_from_labels(lti_out.loss, labels)

            # Baseline alignment for original model (self vs self): expected cosine=1, top1=1.
            base_acc.update_logit_alignment(base_out.logits, base_out.logits, batch["attention_mask"])
            manual_acc.update_logit_alignment(base_out.logits, man_out.logits, batch["attention_mask"])
            lti_acc.update_logit_alignment(base_out.logits, lti_out.logits, batch["attention_mask"])

    report = {
        "meta": {
            "hf_model": args.hf_model,
            "conversion_config": args.conversion_config,
            "target_activation": target_act,
            "eval_source": eval_source,
            "wikitext_subset": args.wikitext_subset,
            "wikitext_split": args.wikitext_split if args.wikitext_subset else None,
            "num_texts": len(texts),
            "corpus_char_count": len(corpus_text),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "stride": args.stride,
            "device": str(device),
            "dtype": args.dtype,
            "manual_checkpoint": str(manual_ckpt),
            "lti_checkpoint": str(lti_ckpt),
            "manual_replaced_modules": manual_replaced,
            "lti_replaced_modules": lti_replaced,
            "enable_layernorm_rsqrt": enable_layernorm_rsqrt,
            "layernorm_mode": layernorm_mode,
            "rsqrt_conversion_config": rsqrt_cfg_name if enable_layernorm_rsqrt else None,
            "rsqrt_bank_manifest": str(rsqrt_bank_manifest) if rsqrt_bank_manifest is not None else None,
            "manual_layernorm_replaced_modules": manual_ln_replaced,
            "lti_layernorm_replaced_modules": lti_ln_replaced,
        },
        "original": base_acc.summarize(),
        "mbe_manual": manual_acc.summarize(),
        "mbe_lti": lti_acc.summarize(),
    }

    print("=" * 80)
    print("HF MBE Validation Report")
    print("=" * 80)
    print(json.dumps(report, indent=2))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
