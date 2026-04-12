"""
Stochastic samplers used by the environment.
"""

import math
import random

from server.constants import (
    HEAT_FAIL_CAP,
    HEAT_FAIL_SLOPE,
)


def _sample_load_time_s(data: dict) -> float:
    avg_ms = data["load_avg_ms"]
    var_ms = data.get("load_var_ms", avg_ms * 0.15)

    mu    = math.log(max(avg_ms, 1e-6))
    sigma = math.sqrt(math.log(1.0 + var_ms / (avg_ms ** 2))) if avg_ms > 0 else 0.15
    sigma = max(0.05, min(sigma, 0.6))

    sampled_ms = math.exp(random.gauss(mu, sigma))

    lo_ms, hi_ms = data.get("load_range_ms", (avg_ms * 0.3, avg_ms * 4.0))
    sampled_ms   = max(lo_ms, min(hi_ms, sampled_ms))

    return sampled_ms / 1000.0


def _sample_host_mb(data: dict) -> float:
    avg = data["host_mb"]
    lo, hi = data.get("host_mb_range", (avg * 0.9, avg * 1.3))
    return random.triangular(lo, hi, avg)


def _quality_failure(model_heat: int) -> bool:
    fail_prob = min(HEAT_FAIL_CAP, HEAT_FAIL_SLOPE * model_heat)
    return random.random() >= fail_prob