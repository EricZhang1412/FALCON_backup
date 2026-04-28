from __future__ import annotations

import random

try:
    import numpy as np
except ImportError:
    np = None

import torch


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
