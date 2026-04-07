from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import statistics

# EpisodeResult — populated by the environment at episode end

@dataclass
class EpisodeResult:
    task_name: str
    total_requests: int
    completed_requests: int
    total_reasoning: int
    reasoning_completed: int
    steps_taken: int
    max_steps: int
    load_count: int
    evict_count: int
    oom_errors: int = 0
    idle_steps: int = 0

    # ── NEW FIELDS ────────────────────────────────────────────────────────────
    completion_ages: List[float] = field(default_factory=list)   # age_steps at serve time
    throughput_samples: List[float] = field(default_factory=list) # effective gen_tps per EXECUTE
    overprovision_count: int = 0   # batches served at higher tier than needed


# ── SLA / Latency helpers ─────────────────────────────────────────────────────

SLA_THRESHOLD_STEPS = 40   # requests served within this age are "on time"

def _latency_score(ages: List[float]) -> float:
    """
    0.0–1.0.  Blends SLA compliance (fraction served in time) and
    mean age normalised against the threshold.
    Returns 1.0 if no data (no penalty for empty).
    """
    if not ages:
        return 1.0
    sla_ok = sum(1 for a in ages if a <= SLA_THRESHOLD_STEPS) / len(ages)
    mean_age = statistics.mean(ages)
    age_score = max(0.0, 1.0 - mean_age / (SLA_THRESHOLD_STEPS * 2))
    return 0.60 * sla_ok + 0.40 * age_score


def _throughput_score(samples: List[float]) -> float:
    """
    0.0–1.0.  Mean effective gen_tps normalised against a reference of 20 t/s.
    Below reference degrades linearly; above is capped at 1.0.
    """
    if not samples:
        return 0.5   # neutral if no executes happened
    REFERENCE_TPS = 20.0
    return min(1.0, statistics.mean(samples) / REFERENCE_TPS)


def _overprovision_score(total_executes: int, overprovision_count: int) -> float:
    """
    1.0 when no over-provisioning.  Penalises the fraction of batches that
    used a higher quant tier than the work required.
    """
    if total_executes == 0:
        return 1.0
    ratio = overprovision_count / total_executes
    return max(0.0, 1.0 - ratio * 0.8)   # full over-provision → 0.2 floor


# ── Individual graders (updated weight splits) ───────────────────────────────

def grade_single_load(result: EpisodeResult) -> float:
    total      = result.total_requests
    completion = result.completed_requests / max(1, total)

    extra_loads  = max(0, result.load_count - 1)
    extra_evicts = max(0, result.evict_count - 0)
    stability    = max(0.0, 1.0 - (extra_loads + extra_evicts) * 0.15)

    latency     = _latency_score(result.completion_ages)
    throughput  = _throughput_score(result.throughput_samples)
    overprov    = _overprovision_score(
        result.load_count + result.completed_requests, result.overprovision_count
    )

    score = (
        0.45 * completion
      + 0.25 * stability
      + 0.15 * latency
      + 0.10 * throughput
      + 0.05 * overprov
    )
    return _clamp(score)


def grade_multi_load(result: EpisodeResult) -> float:
    total   = result.total_requests
    total_r = result.total_reasoning

    completion = result.completed_requests / max(1, total)
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    optimal_cutoff = 8
    overage    = max(0, result.steps_taken - optimal_cutoff)
    efficiency = max(0.0, 1.0 - overage / 18.0)

    latency    = _latency_score(result.completion_ages)
    throughput = _throughput_score(result.throughput_samples)
    overprov   = _overprovision_score(
        result.load_count + result.completed_requests, result.overprovision_count
    )

    score = (
        0.35 * completion
      + 0.20 * reasoning
      + 0.15 * efficiency
      + 0.15 * latency
      + 0.10 * throughput
      + 0.05 * overprov
    )
    return _clamp(score)


def grade_quality_limit(result: EpisodeResult) -> float:
    total   = result.total_requests
    total_r = result.total_reasoning

    completion = result.completed_requests / max(1, total)
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    optimal_cutoff = 8
    overage        = max(0, result.steps_taken - optimal_cutoff)
    step_efficiency = max(0.0, 1.0 - overage / 12.0)
    churn          = result.load_count + result.evict_count
    churn_penalty  = min(1.0, churn * 0.08)
    efficiency     = step_efficiency * (1.0 - churn_penalty)

    latency    = _latency_score(result.completion_ages)
    throughput = _throughput_score(result.throughput_samples)
    overprov   = _overprovision_score(
        result.load_count + result.completed_requests, result.overprovision_count
    )

    score = (
        0.35 * completion
      + 0.25 * reasoning
      + 0.15 * latency
      + 0.10 * efficiency
      + 0.10 * throughput
      + 0.05 * overprov
    )
    return _clamp(score)


def grade_ram_pressure(result: EpisodeResult) -> float:
    total   = result.total_requests
    total_r = result.total_reasoning

    completion = result.completed_requests / max(1, total)
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    oom_penalty = min(1.0, result.oom_errors * 1.0)
    safety      = max(0.0, 1.0 - oom_penalty)

    latency    = _latency_score(result.completion_ages)
    throughput = _throughput_score(result.throughput_samples)

    score = (
        0.30 * completion
      + 0.30 * safety
      + 0.20 * reasoning
      + 0.15 * latency
      + 0.05 * throughput
    )
    return _clamp(score)


def _clamp(score: float) -> float:
    return round(min(0.99, max(0.01, score)), 2)


GRADERS: Dict[str, callable] = {
    "multi-load":    grade_multi_load,
    "single-load":   grade_single_load,
    "quality-limit": grade_quality_limit,
    "ram-pressure":  grade_ram_pressure,
}


def grade(task_name: str, result: EpisodeResult) -> float:
    prefix = task_name.lower().replace("_", "-")
    grader = GRADERS.get(prefix)
    if grader is None:
        raw = result.completed_requests / max(1, result.total_requests)
        return _clamp(raw)
    return grader(result)