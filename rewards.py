"""
rewards.py
----------
All reward / penalty calculations for ModelFlowEnvironment.

Every public function returns a float delta that the caller should add to
the episode reward.  Side-effects (mutating counters, logging, etc.) remain
in the environment; this module is purely numeric.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Clock / age penalties  (called once per logical "tick")
# ---------------------------------------------------------------------------

def clock_tick_penalty(queue, loaded_models) -> float:
    """
    Per-step background penalty applied every time _clock_tick() fires.

    • Each pending request accrues an age penalty that grows super-linearly.
    • Extra penalty when there are pending requests but no model is loaded.

    Returns
    -------
    float
        Negative reward delta (always <= 0).
    """
    reward = 0.0

    for req in queue:
        penalty_val = 0.001 * (req.age_steps ** 1.2)
        reward -= min(penalty_val, 0.5)

    if len(queue) > 0 and len(loaded_models) == 0:
        reward -= 2.0

    return reward


# ---------------------------------------------------------------------------
# LOAD rewards
# ---------------------------------------------------------------------------

def load_bad_args() -> float:
    """LOAD called without required arguments."""
    return 0.0          # error is set; no numeric penalty beyond missing action


def load_unknown_config() -> float:
    """Requested model/quant combo is not in the roster."""
    return 0.0


def load_already_loaded() -> float:
    """Tried to LOAD a model that is already in memory."""
    return -5.0


def load_oom() -> float:
    """LOAD would exceed available RAM."""
    return -30.0


def load_success(actual_load_s: float, loaded_models_after: int) -> float:
    """
    Reward for a successful LOAD.

    Parameters
    ----------
    actual_load_s        : wall-clock seconds the load took (used as time cost).
    loaded_models_after  : total number of models loaded *after* this LOAD
                           (used to reward diversity).
    """
    reward = 0.0
    reward -= actual_load_s * 1.5          # time cost

    if loaded_models_after >= 2:
        reward += 3.0                      # bonus for running multiple models

    return reward


# ---------------------------------------------------------------------------
# EXECUTE rewards
# ---------------------------------------------------------------------------

def execute_bad_args() -> float:
    """EXECUTE called without required arguments."""
    return -5.0


def execute_not_loaded() -> float:
    """Target model/quant is not currently loaded."""
    return -10.0


def execute_empty_queue() -> float:
    """EXECUTE called but the queue is empty."""
    return -5.0


def execute_no_match(fail_count: int) -> float:
    """
    No request in the queue matched this model/quant.

    The penalty escalates quadratically with repeated failures on the same key.

    Parameters
    ----------
    fail_count : cumulative consecutive failures for this key (>= 1).
    """
    return -(5.0 * (fail_count ** 2))


def execute_runtime_oom() -> float:
    """Peak RAM during execution would exceed hardware limit."""
    return -50.0


def execute_success(matching_requests, tier_multipliers: dict, slot_tier: str) -> float:
    """
    Reward for successfully executing a batch of requests.

    Parameters
    ----------
    matching_requests : iterable of RequestInfo objects that were served.
    tier_multipliers  : mapping from tier name to reward multiplier.
    slot_tier         : the tier string of the model slot used.
    """
    reward = 0.0
    multiplier = tier_multipliers.get(slot_tier, 1.0)

    for req in matching_requests:
        gain = 25.0 if req.complexity == "reasoning" else 15.0
        reward += gain * multiplier

    return reward


# ---------------------------------------------------------------------------
# EVICT rewards
# ---------------------------------------------------------------------------

def evict_success(size_mb: float, model_still_needed: bool) -> float:
    """
    Reward for successfully evicting a model.

    Parameters
    ----------
    size_mb            : RAM freed (informational; not currently used in math).
    model_still_needed : True when the evicted model is still required by
                         pending requests (penalised).
    """
    reward = -10.0                         # base eviction cost

    if not model_still_needed:
        reward += 5.0                      # bonus for evicting something truly idle

    return reward


def evict_nothing_to_evict() -> float:
    """EVICT called but nothing matched."""
    return -5.0


# ---------------------------------------------------------------------------
# IDLE rewards
# ---------------------------------------------------------------------------

def idle_penalty() -> float:
    """Flat penalty for choosing IDLE."""
    return -15.0


# ---------------------------------------------------------------------------
# REPLACE rewards
# ---------------------------------------------------------------------------

def replace_no_target() -> float:
    """REPLACE could not identify a model to evict."""
    return -5.0


def replace_evict_component(evicted_model_still_needed: bool) -> float:
    """
    Reward delta for the eviction half of a REPLACE.

    Parameters
    ----------
    evicted_model_still_needed : True when the evicted model is still required.
    """
    if not evicted_model_still_needed:
        return 2.0
    return -15.0


def replace_bad_load_args() -> float:
    """REPLACE load half has missing model_id / quant_type."""
    return -5.0


def replace_load_unknown_config() -> float:
    """New model/quant combo is not in the roster."""
    return -5.0


def replace_load_already_loaded() -> float:
    """Target model is already in memory."""
    return -10.0


def replace_load_oom() -> float:
    """New model would exceed available RAM after eviction."""
    return -30.0


def replace_load_success(
    actual_load_s: float,
    loaded_count_before: int,
    loaded_count_after: int,
) -> float:
    """
    Reward for the successful load half of a REPLACE.

    Parameters
    ----------
    actual_load_s       : seconds taken to load the new model.
    loaded_count_before : number of models loaded *before* the REPLACE started.
    loaded_count_after  : number of models loaded *after* the new model is live.
    """
    reward = 5.0                           # base bonus for a clean swap
    reward -= actual_load_s * 1.5         # time cost

    if loaded_count_before < 2 and loaded_count_after >= 2:
        reward += 1.5                      # reached multi-model milestone

    return reward


# ---------------------------------------------------------------------------
# Episode-terminal rewards
# ---------------------------------------------------------------------------

def episode_success() -> float:
    """All requests were completed – queue is empty."""
    return 50.0


def episode_timeout() -> float:
    """MAX_STEPS reached without clearing the queue."""
    return -50.0