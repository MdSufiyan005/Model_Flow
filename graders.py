from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import statistics


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
    completion_ages: List[float] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)
    overprovision_count: int = 0

    # ── V2 additions ──────────────────────────────────────────────────────────
    # Number of EXECUTE calls where heat caused a quality failure.
    quality_failures: int = 0
    # Deferred requests that were eventually served (DEFER was strategic).
    deferred_served: int = 0
    # Deferred requests that were never served before episode end (wasted defers).
    deferred_abandoned: int = 0
    # Per-request SLA window at serve time (tracks tightening across episode).
    sla_at_serve: List[int] = field(default_factory=list)
    # Age of each request at serve time — parallel list to completion_ages.
    # (completion_ages already exists; sla_at_serve is the threshold at that moment)


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _latency_score(ages: List[float], sla_windows: List[int] = None) -> float:
    """
    V2: uses per-request SLA window if available (tightening SLA tasks).
    Falls back to a fixed SLA_THRESHOLD_STEPS=40 when sla_windows is absent.
    """
    if not ages:
        return 1.0

    if sla_windows and len(sla_windows) == len(ages):
        sla_ok = sum(
            1 for age, sla in zip(ages, sla_windows) if age <= sla
        ) / len(ages)
        # Normalise mean age against the average SLA window seen.
        mean_sla = statistics.mean(sla_windows)
        mean_age = statistics.mean(ages)
        age_score = max(0.0, 1.0 - mean_age / (mean_sla * 2))
    else:
        SLA_THRESHOLD_STEPS = 40
        sla_ok = sum(1 for a in ages if a <= SLA_THRESHOLD_STEPS) / len(ages)
        mean_age = statistics.mean(ages)
        age_score = max(0.0, 1.0 - mean_age / (SLA_THRESHOLD_STEPS * 2))

    return 0.60 * sla_ok + 0.40 * age_score


def _throughput_score(samples: List[float]) -> float:
    if not samples:
        return 0.5
    REFERENCE_TPS = 20.0
    return min(1.0, statistics.mean(samples) / REFERENCE_TPS)


def _overprovision_score(total_executes: int, overprovision_count: int) -> float:
    if total_executes == 0:
        return 1.0
    ratio = overprovision_count / total_executes
    return max(0.0, 1.0 - ratio * 0.8)


def _quality_accuracy_score(completed: int, quality_failures: int) -> float:
    """
    V2: fraction of executions that did NOT suffer a heat-induced quality failure.
    1.0 = no failures, 0.0 = every execution degraded.
    """
    if completed == 0:
        return 1.0
    return max(0.0, 1.0 - quality_failures / completed)


def _defer_efficiency_score(deferred_served: int, deferred_abandoned: int) -> float:
    """
    V2: measures how effective the agent's DEFER decisions were.
    1.0 = every deferred request was eventually served.
    0.0 = every deferred request was abandoned at episode end.
    Returns 1.0 when no defers were issued (no penalty for not deferring).
    """
    total_deferred = deferred_served + deferred_abandoned
    if total_deferred == 0:
        return 1.0
    return deferred_served / total_deferred


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------

def grade_single_load(result: EpisodeResult) -> float:
    """
    Easy task — stable demand, no drift, no shift.
    Tests: load once, execute to completion, avoid churn.
    Quality accuracy added but weighted lightly (minimal heat expected).
    """
    completion = result.completed_requests / max(1, result.total_requests)

    extra_loads  = max(0, result.load_count - 1)
    extra_evicts = max(0, result.evict_count)
    stability    = max(0.0, 1.0 - (extra_loads + extra_evicts) * 0.15)

    latency    = _latency_score(result.completion_ages, result.sla_at_serve)
    throughput = _throughput_score(result.throughput_samples)
    overprov   = _overprovision_score(
        result.load_count + result.completed_requests, result.overprovision_count
    )
    quality_acc = _quality_accuracy_score(result.completed_requests, result.quality_failures)

    score = (
        0.40 * completion
      + 0.22 * stability
      + 0.14 * latency
      + 0.10 * throughput
      + 0.08 * quality_acc
      + 0.06 * overprov
    )
    return _clamp(score)


def grade_multi_load(result: EpisodeResult) -> float:
    """
    Medium task — demand shift at T_shift, load time jitter.
    Tests: detect shift, adapt model choice, manage step count.
    Defer efficiency added: agent may DEFER requests to wait for the right model.
    """
    completion = result.completed_requests / max(1, result.total_requests)
    total_r    = result.total_reasoning
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    optimal_cutoff = 8
    overage    = max(0, result.steps_taken - optimal_cutoff)
    efficiency = max(0.0, 1.0 - overage / 18.0)

    latency      = _latency_score(result.completion_ages, result.sla_at_serve)
    throughput   = _throughput_score(result.throughput_samples)
    overprov     = _overprovision_score(
        result.load_count + result.completed_requests, result.overprovision_count
    )
    quality_acc  = _quality_accuracy_score(result.completed_requests, result.quality_failures)
    defer_eff    = _defer_efficiency_score(result.deferred_served, result.deferred_abandoned)

    score = (
        0.28 * completion
      + 0.18 * reasoning
      + 0.14 * efficiency
      + 0.13 * latency
      + 0.10 * quality_acc
      + 0.09 * throughput
      + 0.05 * defer_eff
      + 0.03 * overprov
    )
    return _clamp(score)


def grade_quality_limit(result: EpisodeResult) -> float:
    """
    Hard task — tightening SLA, model heat/drift, DEFER is useful.
    Tests: proactive heat management, quality-aware quant selection,
           strategic deferral, adapting to a shrinking SLA window.
    Quality accuracy is now a primary signal (0.25 weight).
    """
    completion = result.completed_requests / max(1, result.total_requests)
    total_r    = result.total_reasoning
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    optimal_cutoff  = 8
    overage         = max(0, result.steps_taken - optimal_cutoff)
    step_efficiency = max(0.0, 1.0 - overage / 12.0)
    churn           = result.load_count + result.evict_count
    churn_penalty   = min(1.0, churn * 0.08)
    efficiency      = step_efficiency * (1.0 - churn_penalty)

    latency     = _latency_score(result.completion_ages, result.sla_at_serve)
    throughput  = _throughput_score(result.throughput_samples)
    overprov    = _overprovision_score(
        result.load_count + result.completed_requests, result.overprovision_count
    )
    quality_acc = _quality_accuracy_score(result.completed_requests, result.quality_failures)
    defer_eff   = _defer_efficiency_score(result.deferred_served, result.deferred_abandoned)

    score = (
        0.25 * completion
      + 0.22 * quality_acc     # primary signal on this task
      + 0.18 * reasoning
      + 0.13 * latency
      + 0.08 * efficiency
      + 0.06 * defer_eff
      + 0.05 * throughput
      + 0.03 * overprov
    )
    return _clamp(score)


def grade_ram_pressure(result: EpisodeResult) -> float:
    """
    Extreme task — compound: tightening SLA + demand shift + RAM spikes + heat.
    Tests: OOM avoidance, heat management under tight RAM, strategic DEFER
           to avoid loading the wrong model when RAM is constrained.
    Safety (OOM avoidance) and quality accuracy are co-equal primary signals.
    """
    completion = result.completed_requests / max(1, result.total_requests)
    total_r    = result.total_reasoning
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    oom_penalty = min(1.0, result.oom_errors * 1.0)
    safety      = max(0.0, 1.0 - oom_penalty)

    latency     = _latency_score(result.completion_ages, result.sla_at_serve)
    throughput  = _throughput_score(result.throughput_samples)
    quality_acc = _quality_accuracy_score(result.completed_requests, result.quality_failures)
    defer_eff   = _defer_efficiency_score(result.deferred_served, result.deferred_abandoned)

    score = (
        0.22 * completion
      + 0.22 * safety
      + 0.18 * quality_acc
      + 0.15 * reasoning
      + 0.10 * defer_eff
      + 0.09 * latency
      + 0.04 * throughput
    )
    return _clamp(score)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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