"""
rewards.py — ModelFlow V2
--------------------------
All reward / penalty calculations for ModelFlowEnvironment.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from server.constants import (
    MAX_BLEU_OBSERVED,
    MAX_ROUGE_OBSERVED,
    REASONING_QUANT_PENALTY,
)



# Clock / age penalties

def clock_tick_penalty(queue, loaded_models) -> float:
    reward = 0.0
    for req in queue:
        reward -= min(0.0005 * req.age_steps, 0.15)
    if queue and not loaded_models:
        reward -= 0.5
    return reward


# LOAD rewards

def load_already_loaded() -> float:
    return -5.0


def load_oom() -> float:
    return -30.0


def load_success(actual_load_s: float, loaded_models_after: int) -> float:
    reward = -actual_load_s * 1.0
    if loaded_models_after >= 2:
        reward += 4.0
    return reward


# EXECUTE rewards

def execute_bad_args() -> float:
    return -5.0


def execute_not_loaded() -> float:
    return -10.0


def execute_empty_queue() -> float:
    return -5.0


def execute_no_match(fail_count: int) -> float:
    """
    Quadratically escalating penalty for repeated no-match on same key.
    Cap at -45 so a persistent mismatch doesn't blow up the episode score.
    """
    return -min(5.0 * (fail_count ** 2), 45.0)


def execute_runtime_oom() -> float:
    return -50.0


def execute_success(
    matching_requests,
    tier_multipliers: dict,
    slot_tier: str,
    quant_type: str,
    roster_data: dict,
    quality_ok: bool,
    current_sla_steps: int,
) -> float:
    reward = 0.0
    multiplier = tier_multipliers.get(slot_tier, 1.0)

    bleu  = roster_data.get("bleu_avg", 0.0)
    rouge = roster_data.get("rouge_l_avg", 0.0)
    quality_base = 0.5 * (bleu / MAX_BLEU_OBSERVED) + 0.5 * (rouge / MAX_ROUGE_OBSERVED)
    quality_base = max(0.0, min(1.0, quality_base))
    quality_factor = quality_base if quality_ok else quality_base * 0.5

    high_quants = {"Q6_K", "Q8_0"}
    all_standard = all(r.complexity != "reasoning" for r in matching_requests)
    overprovisioned = all_standard and quant_type in high_quants

    for req in matching_requests:
        if req.complexity == "reasoning":
            penalty = REASONING_QUANT_PENALTY.get(quant_type, 0.0)
            effective_quality = quality_factor * (1.0 - penalty)
            gain = 30.0 * effective_quality * multiplier
        else:
            gain = 20.0 * quality_factor * multiplier
            if overprovisioned:
                gain -= 2.0

        if req.age_steps > current_sla_steps:
            overage = req.age_steps - current_sla_steps
            sla_penalty = min(overage * 0.5, gain * 0.40)
            gain -= sla_penalty

        reward += max(1.0, gain)

    return reward


def quality_degraded_penalty() -> float:
    return -4.0


def late_sla_penalty(overage_steps: int) -> float:
    return -min(overage_steps * 0.3, 10.0)


# EVICT rewards

def evict_success(size_mb: float, model_still_needed: bool) -> float:
    reward = -6.0
    if not model_still_needed:
        reward += 4.0
    return reward


def evict_nothing_to_evict() -> float:
    return -5.0


# IDLE rewards

def idle_penalty() -> float:
    return -15.0


# REPLACE rewards

def replace_no_target() -> float:
    return -5.0


def replace_evict_component(evicted_model_still_needed: bool) -> float:
    # FIX: reduced from -15 to -6 to align with grader.
    # -15 was over-penalising necessary evictions and conflicting with
    # grade_ram_pressure which rewards completion + safety equally.
    # Agent was learning "REPLACE is always bad" even when RAM-forced.
    return 2.0 if not evicted_model_still_needed else -6.0


def replace_bad_load_args() -> float:
    return -5.0


def replace_load_unknown_config() -> float:
    return -5.0


def replace_load_already_loaded() -> float:
    return -10.0


def replace_load_oom() -> float:
    return -30.0


def replace_load_success(
    actual_load_s: float,
    loaded_count_before: int,
    loaded_count_after: int,
) -> float:
    reward = 5.0
    reward -= actual_load_s * 1.0
    if loaded_count_before < 2 and loaded_count_after >= 2:
        reward += 2.0
    return reward


# DEFER rewards

def defer_penalty(req_age: int) -> float:
    base = -3.0
    age_surcharge = -min(req_age * 0.2, 4.0)
    return base + age_surcharge


def defer_serve_bonus(waited_steps: int, quality_ok: bool) -> float:
    bonus = 5.0 + min(waited_steps * 0.4, 4.0)
    if not quality_ok:
        bonus *= 0.5
    return bonus


# Episode-terminal rewards

def episode_success() -> float:
    return 50.0


def episode_timeout() -> float:
    return -50.0