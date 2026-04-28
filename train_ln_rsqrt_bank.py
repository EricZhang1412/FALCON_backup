"""Train per-LN rsqrt conversion configs listed in a manifest."""
from __future__ import annotations

import argparse
import json

from conversion.runner import run_conversion_training


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-LN rsqrt conversion bank.")
    p.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    p.add_argument("--manual", action="store_true", help="Run manual training")
    p.add_argument("--lti", action="store_true", help="Run LTI training")
    p.add_argument("--budget", type=int, default=64, help="LTI budget")
    p.add_argument("--gamma-lo", type=float, default=0.01, help="LTI gamma lower bound")
    p.add_argument("--gamma-hi", type=float, default=0.5, help="LTI gamma upper bound")
    p.add_argument("--limit", type=int, default=0, help="Only train first N entries (0 means all)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(open(args.manifest, "r", encoding="utf-8").read())
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError(f"No entries found in manifest: {args.manifest}")

    run_manual = args.manual or (not args.manual and not args.lti)
    run_lti = args.lti or (not args.manual and not args.lti)
    limit = args.limit if args.limit > 0 else len(entries)

    for idx, item in enumerate(entries[:limit], start=1):
        cfg_name = str(item["config_name"])
        print(f"\n[{idx}/{limit}] Training config: {cfg_name}")
        if run_manual:
            run_conversion_training(cfg_name, use_lti=False, label=f"{cfg_name}/manual")
        if run_lti:
            run_conversion_training(
                cfg_name,
                use_lti=True,
                lti_budget=args.budget,
                lti_gamma_range=(args.gamma_lo, args.gamma_hi),
                label=f"{cfg_name}/lti",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
