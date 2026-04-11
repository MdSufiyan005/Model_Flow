"""
rewards.py — ModelFlow V2
--------------------------
All reward / penalty calculations for ModelFlowEnvironment.

Design principles
-----------------
- Every public function returns a single float delta.
- No side-effects: counter mutations stay in the environment.
- Calibrated so a correct LOAD→EXECUTE sequence yields net-positive reward
  even when the model is slow (high total_time_s → many internal clock ticks).

Root cause of the -767 / -1157 EXECUTE rewards (now fixed)
-----------------------------------------------------------
The environment's EXECUTE path fires `_clock_tick()` for every simulated
second of inference time (`exec_steps - 1` extra ticks after the first).
Each tick called clock_tick_penalty() which charged:

    per request: -min(0.001 * age_steps^1.2, 0.5)

With a slow model and a large remaining queue:
  - age_steps can be 30–80 before execution even starts
  - 0.001 * 80^1.2 ≈ 0.18 per request per tick
  - 10 remaining requests × 0.18 × 40 exec_ticks = -72 from age alone
  - plus the no-model penalty of -2.0/tick when queue non-empty but
    loaded_models is populated... wait, that branch was NOT the problem.
  - The real killer: super-linear growth. age_steps^1.2 at step 60 = 0.22,
    at step 80 = 0.30, at step 120 = 0.46 — each approaching the 0.5 cap.
    With 10 pending × many exec ticks this explodes to -hundreds.

Fix: make the per-request age penalty linear (exponent 1.0 not 1.2) and
reduce the coefficient so a request aged 30 steps costs 0.015/tick not 0.027.
Cap per-request contribution at 0.15 (was 0.5). The no-model surcharge is
kept but capped at -0.5/tick (was -2.0) since it fires redundantly during
the internal EXECUTE simulation loop when loaded_models is non-empty anyway.

SLA penalty in execute_success also fixed: was capping at 60% of gain
(gain - 60% = only 40% left). Now capped at 40% of gain so the agent
always retains at least 60% of the base reward even for very stale requests.
This prevents execute_success from returning near-zero for aged batches.

Quant overprovisioning: Q6_K for standard requests now costs a small
explicit penalty so the LLM stops using it reflexively on all requests.
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


# ---------------------------------------------------------------------------
# Clock / age penalties
# ---------------------------------------------------------------------------

def clock_tick_penalty(queue, loaded_models) -> float:
    """
    Per-step background penalty applied every time _clock_tick() fires.

    CALIBRATION (vs original):
    - Exponent reduced 1.2 → 1.0  (linear growth, not super-linear)
    - Coefficient reduced 0.001 → 0.0005
    - Per-request cap reduced 0.5 → 0.15
    - No-model surcharge reduced -2.0 → -0.5, and only fires when
      loaded_models is truly empty (unchanged logic, reduced magnitude)

    These values still create urgency to serve requests quickly, but they
    no longer compound to -hundreds inside a multi-tick EXECUTE loop.
    """
    reward = 0.0
    for req in queue:
        reward -= min(0.0005 * req.age_steps, 0.15)
    if queue and not loaded_models:
        reward -= 0.5
    return reward


# ---------------------------------------------------------------------------
# LOAD rewards
# ---------------------------------------------------------------------------

def load_already_loaded() -> float:
    return -5.0


def load_oom() -> float:
    return -30.0


def load_success(actual_load_s: float, loaded_models_after: int) -> float:
    """
    Penalise load time (unavoidable cost of cold start) and reward having
    multiple models ready (amortises future REPLACE overhead).
    Coefficient 1.5 → 1.0: load time penalty was too steep relative to
    the gain from execute_success (~15–25 per request).
    """
    reward = -actual_load_s * 1.0
    if loaded_models_after >= 2:
        reward += 4.0
    return reward


# ---------------------------------------------------------------------------
# EXECUTE rewards
# ---------------------------------------------------------------------------

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
    """
    Reward for successfully executing a batch.

    Changes vs previous version
    ---------------------------
    - SLA overage penalty capped at 40% of gain (was 60%).
      At 60% the net reward for an aged request could approach zero,
      making it indistinguishable from doing nothing — which confused the
      LLM into thinking EXECUTE was a bad action.
    - Gain floor raised: max(1.0, gain) instead of max(0.0, gain).
      Even a badly-aged, quality-degraded request should yield a small
      positive reward to reinforce that serving > not serving.
    - quant_overprovisioning_penalty: if standard-only requests are served
      with Q6_K or higher, subtract a small cost to discourage reflexive
      use of the highest quant on every queue.
    """
    reward = 0.0
    multiplier = tier_multipliers.get(slot_tier, 1.0)

    bleu  = roster_data.get("bleu_avg", 0.0)
    rouge = roster_data.get("rouge_l_avg", 0.0)
    quality_base = 0.5 * (bleu / MAX_BLEU_OBSERVED) + 0.5 * (rouge / MAX_ROUGE_OBSERVED)
    quality_base = max(0.0, min(1.0, quality_base))
    quality_factor = quality_base if quality_ok else quality_base * 0.5

    # Small penalty for using a high quant on standard-only requests.
    # Q6_K on a standard request wastes RAM capacity for no quality gain.
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
                gain -= 2.0   # small cost for unnecessary high quant

        if req.age_steps > current_sla_steps:
            overage = req.age_steps - current_sla_steps
            # Cap SLA penalty at 40% of gross gain (was 60%).
            # Agent always keeps ≥60% of base reward even for stale requests.
            sla_penalty = min(overage * 0.5, gain * 0.40)
            gain -= sla_penalty

        # Floor at 1.0: serving a request always beats not serving it.
        reward += max(1.0, gain)

    return reward


def quality_degraded_penalty() -> float:
    """
    One-off penalty when heat causes a quality failure.
    Stacks on top of the halved quality_factor inside execute_success.
    Reduced -8.0 → -4.0: the halved quality_factor already cuts the
    gain substantially; double-penalising was too harsh.
    """
    return -4.0


def late_sla_penalty(overage_steps: int) -> float:
    """Standalone helper for very-late deferred requests."""
    return -min(overage_steps * 0.3, 10.0)


# ---------------------------------------------------------------------------
# EVICT rewards
# ---------------------------------------------------------------------------

def evict_success(size_mb: float, model_still_needed: bool) -> float:
    """
    Base cost reduced -10 → -6: eviction is a necessary housekeeping action
    and shouldn't be so expensive that the agent avoids it even when RAM is
    critically full. The still_needed penalty remains to discourage evicting
    models that have matching pending requests.
    """
    reward = -6.0
    if not model_still_needed:
        reward += 4.0
    return reward


def evict_nothing_to_evict() -> float:
    return -5.0


# ---------------------------------------------------------------------------
# IDLE rewards
# ---------------------------------------------------------------------------

def idle_penalty() -> float:
    return -15.0


# ---------------------------------------------------------------------------
# REPLACE rewards
# ---------------------------------------------------------------------------

def replace_no_target() -> float:
    return -5.0


def replace_evict_component(evicted_model_still_needed: bool) -> float:
    return 2.0 if not evicted_model_still_needed else -15.0


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
    reward -= actual_load_s * 1.0   # same coefficient as load_success
    if loaded_count_before < 2 and loaded_count_after >= 2:
        reward += 2.0
    return reward


# ---------------------------------------------------------------------------
# DEFER rewards
# ---------------------------------------------------------------------------

def defer_penalty(req_age: int) -> float:
    """
    Immediate cost for deferring. Small but non-zero.
    Older requests get a larger penalty — don't defer what's already stale.
    """
    base = -3.0
    age_surcharge = -min(req_age * 0.2, 4.0)
    return base + age_surcharge


def defer_serve_bonus(waited_steps: int, quality_ok: bool) -> float:
    """
    Bonus when a deferred request is eventually served.
    Grows with wait time up to a cap; halved if quality degraded anyway.
    """
    bonus = 5.0 + min(waited_steps * 0.4, 4.0)
    if not quality_ok:
        bonus *= 0.5
    return bonus


# ---------------------------------------------------------------------------
# Episode-terminal rewards
# ---------------------------------------------------------------------------

def episode_success() -> float:
    return 50.0


def episode_timeout() -> float:
    return -50.0
# """
# rewards.py
# ----------
# All reward / penalty calculations for ModelFlowEnvironment.

# Every public function returns a float delta the caller adds to episode reward.
# Side-effects (mutating counters, logging) stay in the environment.

# V2 additions
# ------------
# - execute_success now factors real BLEU/ROUGE quality scores from benchmark data
# - quality_degraded_penalty  called when a heat-induced quality failure occurs
# - defer_penalty              called when agent issues DEFER
# - defer_serve_bonus          called when a deferred request is eventually served
# - late_sla_penalty           called per request served outside current SLA window
# """

# from __future__ import annotations
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     pass   # avoid circular import; RequestInfo is passed as plain objects

# from server.constants import (
#     MAX_BLEU_OBSERVED,
#     MAX_ROUGE_OBSERVED,
#     REASONING_QUANT_PENALTY,
# )


# # ---------------------------------------------------------------------------
# # Clock / age penalties
# # ---------------------------------------------------------------------------

# def clock_tick_penalty(queue, loaded_models) -> float:
#     """
#     Per-step background penalty applied every time _clock_tick() fires.

#     Each pending request accrues a super-linear age penalty.
#     Extra penalty when there are pending requests but no model is loaded.
#     """
#     reward = 0.0
#     for req in queue:
#         reward -= min(0.001 * (req.age_steps ** 1.2), 0.5)
#     if queue and not loaded_models:
#         reward -= 2.0
#     return reward


# # ---------------------------------------------------------------------------
# # LOAD rewards
# # ---------------------------------------------------------------------------

# def load_already_loaded() -> float:
#     return -5.0


# def load_oom() -> float:
#     return -30.0


# def load_success(actual_load_s: float, loaded_models_after: int) -> float:
#     reward = -actual_load_s * 1.5
#     if loaded_models_after >= 2:
#         reward += 3.0
#     return reward


# # ---------------------------------------------------------------------------
# # EXECUTE rewards
# # ---------------------------------------------------------------------------

# def execute_bad_args() -> float:
#     return -5.0


# def execute_not_loaded() -> float:
#     return -10.0


# def execute_empty_queue() -> float:
#     return -5.0


# def execute_no_match(fail_count: int) -> float:
#     return -(5.0 * (fail_count ** 2))


# def execute_runtime_oom() -> float:
#     return -50.0


# def execute_success(
#     matching_requests,
#     tier_multipliers: dict,
#     slot_tier: str,
#     quant_type: str,
#     roster_data: dict,
#     quality_ok: bool,
#     current_sla_steps: int,
# ) -> float:
#     """
#     Reward for successfully executing a batch of requests.

#     V2 changes vs V1
#     ----------------
#     - Base gain is now anchored to the model's real BLEU/ROUGE scores from
#       benchmark data, normalised against the max observed across all 12 profiles.
#     - Reasoning requests get an additional quality penalty if a low quant tier
#       is used, derived from measured perplexity sensitivity.
#     - quality_ok=False (heat-induced degradation) halves the quality component.
#     - Requests served past current_sla_steps incur a late penalty.

#     Parameters
#     ----------
#     matching_requests  : iterable of RequestInfo that were served.
#     tier_multipliers   : mapping tier → throughput multiplier (unchanged from V1).
#     slot_tier          : tier of the model slot used.
#     quant_type         : quant string e.g. "Q6_K" — used for reasoning penalty.
#     roster_data        : the benchmark dict for this model/quant key.
#     quality_ok         : False when heat caused a quality failure this execution.
#     current_sla_steps  : current SLA window (tightens over time on hard tasks).
#     """
#     reward = 0.0
#     multiplier = tier_multipliers.get(slot_tier, 1.0)

#     # ── quality score grounded in real benchmark measurements ──────────────
#     bleu  = roster_data.get("bleu_avg",   0.0)
#     rouge = roster_data.get("rouge_l_avg", 0.0)
#     # Normalise to [0, 1] against the best observed values in the dataset.
#     quality_base = 0.5 * (bleu / MAX_BLEU_OBSERVED) + 0.5 * (rouge / MAX_ROUGE_OBSERVED)
#     quality_base = max(0.0, min(1.0, quality_base))

#     # Heat degradation: quality failure halves the quality component.
#     quality_factor = quality_base if quality_ok else quality_base * 0.5

#     for req in matching_requests:
#         if req.complexity == "reasoning":
#             # Penalise using a low quant for reasoning — grounded in perplexity data.
#             penalty = REASONING_QUANT_PENALTY.get(quant_type, 0.0)
#             effective_quality = quality_factor * (1.0 - penalty)
#             gain = 30.0 * effective_quality * multiplier
#         else:
#             gain = 20.0 * quality_factor * multiplier

#         # SLA penalty: linear decay for each step over the window.
#         if req.age_steps > current_sla_steps:
#             overage = req.age_steps - current_sla_steps
#             sla_penalty = min(overage * 0.5, gain * 0.6)   # caps at 60% of gain
#             gain -= sla_penalty

#         reward += max(0.0, gain)

#     return reward


# # ---------------------------------------------------------------------------
# # Quality degradation (heat-induced failure, called separately from env)
# # ---------------------------------------------------------------------------

# def quality_degraded_penalty() -> float:
#     """
#     Additional one-off penalty when heat causes a quality failure.
#     Stacks on top of the halved gain inside execute_success.
#     """
#     return -8.0


# # ---------------------------------------------------------------------------
# # Late SLA penalty (standalone, for deferred requests served very late)
# # ---------------------------------------------------------------------------

# def late_sla_penalty(overage_steps: int) -> float:
#     """
#     Extra penalty for a deferred request served well past the SLA window.
#     Called only when overage is large enough to warrant a separate signal.
#     """
#     return -min(overage_steps * 0.3, 10.0)


# # ---------------------------------------------------------------------------
# # EVICT rewards
# # ---------------------------------------------------------------------------

# def evict_success(size_mb: float, model_still_needed: bool) -> float:
#     reward = -10.0
#     if not model_still_needed:
#         reward += 5.0
#     return reward


# def evict_nothing_to_evict() -> float:
#     return -5.0


# # ---------------------------------------------------------------------------
# # IDLE rewards
# # ---------------------------------------------------------------------------

# def idle_penalty() -> float:
#     return -15.0


# # ---------------------------------------------------------------------------
# # REPLACE rewards
# # ---------------------------------------------------------------------------

# def replace_no_target() -> float:
#     return -5.0


# def replace_evict_component(evicted_model_still_needed: bool) -> float:
#     return 2.0 if not evicted_model_still_needed else -15.0


# def replace_bad_load_args() -> float:
#     return -5.0


# def replace_load_unknown_config() -> float:
#     return -5.0


# def replace_load_already_loaded() -> float:
#     return -10.0


# def replace_load_oom() -> float:
#     return -30.0


# def replace_load_success(
#     actual_load_s: float,
#     loaded_count_before: int,
#     loaded_count_after: int,
# ) -> float:
#     reward = 5.0
#     reward -= actual_load_s * 1.5
#     if loaded_count_before < 2 and loaded_count_after >= 2:
#         reward += 1.5
#     return reward


# # ---------------------------------------------------------------------------
# # DEFER rewards
# # ---------------------------------------------------------------------------

# def defer_penalty(req_age: int) -> float:
#     """
#     Immediate cost for deferring a request.

#     Deferring is sometimes correct (wait for better quant / RAM pressure clears).
#     The cost is small but non-zero so the agent doesn't defer reflexively.
#     Older requests incur a larger penalty — you shouldn't defer something
#     that's already been waiting a long time.
#     """
#     base = -3.0
#     age_surcharge = -min(req_age * 0.2, 4.0)   # caps at -4.0 for very old requests
#     return base + age_surcharge


# def defer_serve_bonus(waited_steps: int, quality_ok: bool) -> float:
#     """
#     Bonus when a previously deferred request is successfully served.

#     Rewards the agent for strategically deferring and then serving at higher quality.
#     The bonus grows slightly with wait time (up to a cap) to reward planning ahead,
#     but is halved if quality still degraded despite the wait.
#     """
#     bonus = 5.0 + min(waited_steps * 0.4, 4.0)   # max bonus ≈ 9.0
#     if not quality_ok:
#         bonus *= 0.5
#     return bonus


# # ---------------------------------------------------------------------------
# # Episode-terminal rewards
# # ---------------------------------------------------------------------------

# def episode_success() -> float:
#     return 50.0


# def episode_timeout() -> float:
#     return -50.0