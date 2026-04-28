from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from configs.base import FALLBACK_DEVICE, FALLBACK_DTYPE, FALLBACK_SEED, FALLBACK_T, coerce_dtype

YAML_ROOT = Path(__file__).resolve().parent / "yaml"


@lru_cache(maxsize=None)
def _read_yaml_file(file_path: str) -> dict[str, Any]:
    with Path(file_path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    return deepcopy(_read_yaml_file(str(path)))


def load_project_defaults() -> dict[str, Any]:
    path = YAML_ROOT / "project" / "defaults.yaml"
    payload = _load_yaml(path)
    runtime = payload.get("runtime", {})
    return {
        "seed": int(payload.get("seed", FALLBACK_SEED)),
        "runtime": {
            "dtype": coerce_dtype(runtime.get("dtype"), FALLBACK_DTYPE),
            "device": str(runtime.get("device", FALLBACK_DEVICE)),
            "default_T": int(runtime.get("default_T", FALLBACK_T)),
        },
    }


def load_training_spec(name: str) -> dict[str, Any]:
    path = YAML_ROOT / "training" / f"{name}.yaml"
    return _load_yaml(path)


_PROJECT_DEFAULTS = load_project_defaults()
DEFAULT_SEED = _PROJECT_DEFAULTS["seed"]
DEFAULT_DTYPE = _PROJECT_DEFAULTS["runtime"]["dtype"]
DEFAULT_DEVICE = _PROJECT_DEFAULTS["runtime"]["device"]
