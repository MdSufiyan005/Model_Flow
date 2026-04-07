from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

'''
i think whether the whole eposide is not or not should also be given peanelized
'''
# ---------------------------------------------------------------------------
# EpisodeResult — populated by the environment at episode end
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    task_name: str

    # Request counts
    total_requests: int          # total requests in this task
    completed_requests: int      # requests successfully served

    # Reasoning-specific
    total_reasoning: int         # requests flagged reasoning=True
    reasoning_completed: int     # reasoning requests successfully served

    # Step budget
    steps_taken: int
    max_steps: int               # e.g. 1800 (time budget)

    # Efficiency signals
    load_count: int              # number of LOAD actions executed
    evict_count: int             # number of EVICT actions executed
    oom_errors: int = 0          # OOM penalty hits
    idle_steps: int = 0          # IDLE actions taken with non-empty queue


# ---------------------------------------------------------------------------
# Strict clamp — scores must be strictly between 0 and 1 (exclusive)
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp score to the open interval (0.001, 0.999)."""
    return round(min(0.999, max(0.001, score)), 3)


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------

def grade_multi_load(result: EpisodeResult) -> float:
    """
    EASY task — 12 mixed requests (4 reasoning).
    Tests co-residency of chatbot + coder + translator models.
    
    Weights: 50% completion, 30% reasoning coverage, 20% efficiency.
    """
    total = result.total_requests      # 12
    total_r = result.total_reasoning   # 4

    completion = result.completed_requests / max(1, total)
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    # Efficiency: optimal ≈ 8 steps for this easy task
    optimal_cutoff = 8
    overage = max(0, result.steps_taken - optimal_cutoff)
    efficiency = max(0.0, 1.0 - overage / 18.0)

    score = 0.50 * completion + 0.30 * reasoning + 0.20 * efficiency
    return _clamp(score)


def grade_single_load(result: EpisodeResult) -> float:
    """
    MEDIUM task — 18 identical chatbot requests.
    Tests stability / no-thrash policy (minimize model switching).
    
    Weights: 60% completion, 40% stability.
    """
    total = result.total_requests      # 18
    completion = result.completed_requests / max(1, total)

    # Stability: Optimal = 1 LOAD, 0 EVICTs
    extra_loads  = max(0, result.load_count - 1)
    extra_evicts = max(0, result.evict_count - 0)
    stability = max(0.0, 1.0 - (extra_loads + extra_evicts) * 0.15)

    score = 0.60 * completion + 0.40 * stability
    return _clamp(score)


def grade_quality_limit(result: EpisodeResult) -> float:
    """
    HARD task — 14 mixed requests (4 reasoning).
    Tests reasoning coverage, correct quant selection, and stable residency
    under SLA pressure (minimize churn).
    
    Weights: 50% completion, 40% reasoning coverage, 10% efficiency.
    """
    total = result.total_requests      # 14
    total_r = result.total_reasoning   # 4

    completion = result.completed_requests / max(1, total)
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    # Step efficiency (optimal ≈ 8 steps)
    optimal_cutoff = 8
    overage = max(0, result.steps_taken - optimal_cutoff)
    step_efficiency = max(0.0, 1.0 - overage / 12.0)

    # Churn penalty — Quality Limit demands stability
    # (each LOAD or EVICT is expensive and disruptive)
    churn = result.load_count + result.evict_count
    churn_penalty = min(1.0, churn * 0.08)          # 4 churn actions = 0.32 penalty

    efficiency = step_efficiency * (1.0 - churn_penalty)

    score = 0.50 * completion + 0.40 * reasoning + 0.10 * efficiency
    return _clamp(score)


def grade_ram_pressure(result: EpisodeResult) -> float:
    """
    EXTRA HARD task — 10 mixed requests (4 reasoning).
    Tests safety and robustness during RAM pressure spikes.
    
    Weights: 40% completion, 40% OOM safety, 20% reasoning coverage.
    """
    total = result.total_requests      # 10
    total_r = result.total_reasoning   # 4

    completion = result.completed_requests / max(1, total)
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    # OOM Safety: Each OOM is a severe penalty
    oom_penalty = min(1.0, result.oom_errors * 1.0)
    safety = max(0.0, 1.0 - oom_penalty)

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
    """
    Main entry point. Returns a score strictly in (0.001, 0.999).
    """
    prefix = task_name.lower().replace("_", "-")   # standardize task names
    grader = GRADERS.get(prefix)

    if grader is None:
        # Emergency fallback (should never be hit in normal use)
        raw = result.completed_requests / max(1, result.total_requests)
        return _clamp(raw)

    return grader(result)