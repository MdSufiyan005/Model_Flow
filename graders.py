from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

# EpisodeResult — populated by the environment at episode end

@dataclass
class EpisodeResult:
    task_name: str

    # Request counts
    total_requests: int
    completed_requests: int

    # Reasoning-specific
    total_reasoning: int
    reasoning_completed: int

    # Step budget
    steps_taken: int
    max_steps: int

    # Efficiency signals
    load_count: int
    evict_count: int
    oom_errors: int = 0
    idle_steps: int = 0


# ---------------------------------------------------------------------------
# Strict clamp — scores must be STRICTLY between 0 and 1 (exclusive).
# The 0.995 cap on each component prevents the weighted sum from ever
# hitting exactly 1.0; the 0.005 floor prevents exactly 0.0.
# _clamp is an additional final safety net.
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    return round(min(0.999, max(0.001, score)), 3)


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------

def grade_multi_load(result: EpisodeResult) -> float:
    """EASY — mixed requests (4 reasoning). Weights: 50/30/20."""
    total   = result.total_requests
    total_r = result.total_reasoning

    completion = min(result.completed_requests / max(1, total), 0.995)
    reasoning  = min(result.reasoning_completed / max(1, total_r), 0.995) if total_r > 0 else 0.995

    optimal_cutoff = 8
    overage    = max(0, result.steps_taken - optimal_cutoff)
    efficiency = max(0.005, 1.0 - overage / 18.0)

    score = 0.50 * completion + 0.30 * reasoning + 0.20 * efficiency
    return _clamp(score)


def grade_single_load(result: EpisodeResult) -> float:
    """MEDIUM — identical chatbot requests. Weights: 60/40."""
    total = result.total_requests

    completion = min(result.completed_requests / max(1, total), 0.995)

    extra_loads  = max(0, result.load_count  - 1)
    extra_evicts = max(0, result.evict_count - 0)
    stability    = max(0.005, 1.0 - (extra_loads + extra_evicts) * 0.15)

    score = 0.60 * completion + 0.40 * stability
    return _clamp(score)


def grade_quality_limit(result: EpisodeResult) -> float:
    """HARD — mixed requests (4 reasoning). Weights: 50/40/10."""
    total   = result.total_requests
    total_r = result.total_reasoning

    completion = min(result.completed_requests / max(1, total), 0.995)
    reasoning  = min(result.reasoning_completed / max(1, total_r), 0.995) if total_r > 0 else 0.995

    optimal_cutoff  = 8
    overage         = max(0, result.steps_taken - optimal_cutoff)
    step_efficiency = max(0.005, 1.0 - overage / 12.0)

    churn         = result.load_count + result.evict_count
    churn_penalty = min(0.995, churn * 0.08)
    efficiency    = step_efficiency * (1.0 - churn_penalty)

    score = 0.50 * completion + 0.40 * reasoning + 0.10 * efficiency
    return _clamp(score)


def grade_ram_pressure(result: EpisodeResult) -> float:
    """EXTRA HARD — mixed requests (4 reasoning). Weights: 40/40/20."""
    total   = result.total_requests
    total_r = result.total_reasoning

    completion = min(result.completed_requests / max(1, total), 0.995)
    reasoning  = min(result.reasoning_completed / max(1, total_r), 0.995) if total_r > 0 else 0.995

    oom_penalty = min(0.995, result.oom_errors * 1.0)
    safety      = max(0.005, 1.0 - oom_penalty)

    score = 0.40 * completion + 0.40 * safety + 0.20 * reasoning
    return _clamp(score)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GRADERS: Dict[str, callable] = {
    "multi-load":    grade_multi_load,
    "single-load":   grade_single_load,
    "quality-limit": grade_quality_limit,
    "ram-pressure":  grade_ram_pressure,
}


def grade(task_name: str, result: EpisodeResult) -> float:
    """Returns a score strictly in (0.001, 0.999)."""
    prefix = task_name.lower().replace("_", "-")
    grader = GRADERS.get(prefix)

    if grader is None:
        raw = result.completed_requests / max(1, result.total_requests)
        return _clamp(raw)

    return grader(result)