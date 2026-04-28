from __future__ import annotations

import math
from models.interfaces import BasisSchedule


def safe_log(value: float, floor: float = 1e-300) -> float:
    return math.log(value if value > floor else floor)


def stable_log_ratio(numerator: float, denominator: float, floor: float = 1e-300) -> float:
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}.")
    return safe_log(numerator, floor=floor) - safe_log(denominator, floor=floor)


def strict_crosses_threshold(level: float, schedule: BasisSchedule, t: int) -> bool:
    return schedule.crosses_threshold(level, t)


def first_spike_time_from(level, schedule, start_t, T):
    for t in range(start_t, T):
        if strict_crosses_threshold(level, schedule, t):
            return t
    return None


def candidate_spike_time(level, schedule, t_prev, T):
    directed_level = level if schedule.polarity == "positive" else -level
    if directed_level <= 0:
        return None, None
    raw = stable_log_ratio(directed_level, schedule.alpha_v) / safe_log(schedule.lambda_v)
    candidate = max(t_prev + 1, math.floor(raw) + 1)
    if candidate > T - 1:
        return None, raw
    return candidate, raw


def verify_candidate_time(level, schedule, candidate_t, t_prev, T, window=2):
    if candidate_t is None:
        return None
    lower = max(t_prev + 1, candidate_t - window)
    upper = min(T - 1, candidate_t + window)
    for t in range(lower, upper + 1):
        if strict_crosses_threshold(level, schedule, t):
            return t
    return first_spike_time_from(level, schedule, upper + 1, T)


def decode_spike_time(level, schedule, t_prev, T, verify_candidates=True, candidate_window=2):
    candidate_t, raw = candidate_spike_time(level, schedule, t_prev, T)
    if not verify_candidates:
        return candidate_t, raw, candidate_t
    verified_t = verify_candidate_time(level, schedule, candidate_t, t_prev, T, window=candidate_window)
    return candidate_t, raw, verified_t
