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
    quality_failures: int = 0
    deferred_served: int = 0
    deferred_abandoned: int = 0
    sla_at_serve: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-score helpers  (unchanged from V2)
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
    """
    if completed == 0:
        return 1.0
    return max(0.0, 1.0 - quality_failures / completed)


def _defer_efficiency_score(deferred_served: int, deferred_abandoned: int) -> float:
    """
    V2: measures how effective the agent's DEFER decisions were.
    """
    total_deferred = deferred_served + deferred_abandoned
    if total_deferred == 0:
        return 1.0
    return deferred_served / total_deferred


# ---------------------------------------------------------------------------
# OOM safety  — multiplicative ceiling, not an additive term
#
# A GPU process OOM is a hard failure: in-flight requests are dropped and
# the model must be reloaded from scratch.  Treating OOM as one weighted
# term among many lets other high sub-scores absorb the hit — that is why
# the old grader gave 0.99 on runs with OOM errors.
#
# Instead we apply a score *ceiling* that shrinks with each OOM event.
# The ceiling halves the remaining headroom above 0.15 per OOM:
#   0 OOMs → ceiling 1.000  (no cap)
#   1 OOM  → ceiling 0.618  (~mid-range maximum)
#   2 OOMs → ceiling 0.407
#   3 OOMs → ceiling 0.291
# This makes it structurally impossible for other sub-scores to compensate.
# ---------------------------------------------------------------------------

def _oom_ceiling(oom_errors: int) -> float:
    if oom_errors == 0:
        return 1.0
    ceiling = 1.0
    for _ in range(oom_errors):
        ceiling = 0.15 + (ceiling - 0.15) * 0.55
    return round(ceiling, 3)


# ---------------------------------------------------------------------------
# Churn score  — used only by grade_ram_pressure
#
# Each LOAD + EVICT + REPLACE on a RAM-constrained system costs real
# wall-clock time (model deserialise + PCIe transfer + KV-cache warmup).
# Two events is the minimum viable sequence for this task.
# Each event above 2 costs 0.13 score points.
#
# Calibration:
#   2 events  → 1.00   (perfect memory planning)
#   4 events  → 0.74   (one extra swap, acceptable)
#   6 events  → 0.48   (two extra swaps, noticeable cost)
#   7 events  → 0.35   (meaningful penalty for 7 swaps)
#   9 events  → 0.09   (excessive churn)
#   10+ events → 0.00
# ---------------------------------------------------------------------------

def _churn_score(load_count: int, evict_count: int) -> float:
    churn = load_count + evict_count
    return max(0.0, 1.0 - max(0, churn - 2) * 0.13)


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------

def grade_single_load(result: EpisodeResult) -> float:
    """
    Easy task — stable demand, no drift, no shift.
    Tests: load once, execute to completion, avoid churn.
    Unchanged from V2.
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
    Unchanged from V2.
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

    V3 change: OOM errors now apply a multiplicative ceiling via
    _oom_ceiling().  An OOM here means the agent attempted a quant that
    exceeded the available budget — a memory planning failure.  The
    per-step reward already fired at -45; the ceiling prevents other
    high sub-scores from absorbing that signal at the grader level.
    All weights unchanged from V2.
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
      + 0.22 * quality_acc
      + 0.18 * reasoning
      + 0.13 * latency
      + 0.08 * efficiency
      + 0.06 * defer_eff
      + 0.05 * throughput
      + 0.03 * overprov
    )

    # OOM ceiling — consistent with grade_ram_pressure.
    score = min(score, _oom_ceiling(result.oom_errors))

    return _clamp(score)


def grade_ram_pressure(result: EpisodeResult) -> float:
    """
    Extreme task — compound: tightening SLA + demand shift + RAM spikes + heat.
    Tests: OOM avoidance, heat management under tight RAM, strategic DEFER.

    ── Why the old grader gave 0.99 for bad runs ────────────────────────────
    The old design had `safety = max(0, 1 - oom_errors)` as one additive
    term (weight 0.22).  On a run with 1 OOM + success=False, that term
    zeroed out, but the other six terms — all near 1.0 for a run that
    happened to complete its queue — kept the weighted sum at ~0.78+.
    The score never actually reflected the operational severity of an OOM.

    ── V3 design principles ─────────────────────────────────────────────────

    1. OOM is a hard score ceiling via _oom_ceiling(), not an additive term.
       0 OOMs → no cap.  1 OOM → max ~0.62.  2 OOMs → max ~0.41.
       Structurally impossible for other high sub-scores to compensate.

    2. Queue-not-cleared is a hard multiplier (×0.75).
       success=False means user-visible failures at the service level.
       Completing 90% of requests is not a 90%-good operational outcome.

    3. Churn is a PRIMARY weighted term (weight 0.18) via _churn_score().
       In a real inference cluster each model swap costs wall-clock load
       time.  _churn_score() maps: 2 events→1.0, 7 events→0.35, 10+→0.0.
       This is the mechanism that differentiates good from mediocre runs
       on this task — even a successful run with 7 swaps gets 0.35 on
       this dimension, pulling the weighted sum down meaningfully.

    4. Step soft multiplier threshold raised from 12→16, decay 4%/step.
       A clean 14-step run no longer receives a 32% haircut.  The hard
       task legitimately needs more steps to stay within RAM budget, and
       taking an extra step to avoid an OOM is the correct tradeoff.

    5. Safety term removed from weighted sum; weight redistributed to
       churn_score (0.18) and completion (+0.04).

    ── Expected score ranges ────────────────────────────────────────────────
      Ideal     (8 steps,  2 churn,  0 OOM, success=T):  ~0.95–0.99
      Good      (10 steps, 4 churn,  0 OOM, success=T):  ~0.88–0.95
      Acceptable (14 steps, 7 churn, 0 OOM, success=T):  ~0.83–0.90
      OOM once  (15 steps, 6 churn,  1 OOM, success=F):  ~0.40–0.50
      OOM twice (15 steps, 7 churn,  2 OOM, success=F):  ~0.28–0.35
      Pathology (20 steps, 12 churn, 0 OOM, success=T):  ~0.60–0.70
    """
    completion = result.completed_requests / max(1, result.total_requests)
    total_r    = result.total_reasoning
    reasoning  = result.reasoning_completed / max(1, total_r) if total_r > 0 else 1.0

    latency     = _latency_score(result.completion_ages, result.sla_at_serve)
    throughput  = _throughput_score(result.throughput_samples)
    quality_acc = _quality_accuracy_score(result.completed_requests, result.quality_failures)
    defer_eff   = _defer_efficiency_score(result.deferred_served, result.deferred_abandoned)
    churn_sc    = _churn_score(result.load_count, result.evict_count)

    # safety removed as additive term — enforced as hard ceiling below.
    score = (
        0.22 * completion
      + 0.20 * quality_acc
      + 0.18 * churn_sc      # primary: wall-clock cost of model swaps
      + 0.16 * reasoning
      + 0.12 * latency
      + 0.08 * defer_eff
      + 0.04 * throughput
    )

    # ── Hard ceiling: OOM errors ─────────────────────────────────────────────
    # One OOM caps the score at ~0.62.  Two cap it at ~0.41.
    # This is what prevented the old grader from being realistic.
    score = min(score, _oom_ceiling(result.oom_errors))

    # ── Hard multiplier: queue not cleared ───────────────────────────────────
    queue_cleared = (
        result.completed_requests >= result.total_requests
        and result.deferred_abandoned == 0
    )
    if not queue_cleared:
        score *= 0.75

    # ── Soft multiplier: excessive steps ─────────────────────────────────────
    # Kicks in at step 17 (was 13).  Decay 4%/step (was 8%).
    if result.steps_taken > 16:
        step_factor = max(0.70, 1.0 - (result.steps_taken - 16) * 0.04)
        score *= step_factor

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
