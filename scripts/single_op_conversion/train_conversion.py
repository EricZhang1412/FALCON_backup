"""Run conversion training: python -m scripts.single_op_conversion.train_conversion [config_name] [--lti]"""
from __future__ import annotations

import sys
from falcon.conversion.runner import run_conversion_training


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    config_name = "gelu_conversion"
    use_lti = False
    for a in args:
        if a == "--lti":
            use_lti = True
        elif not a.startswith("--"):
            config_name = a
    run_conversion_training(config_name, use_lti=use_lti)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
