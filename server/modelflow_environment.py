"""
modelflow_environment.py

"""

import math
import os
import random
from typing import Dict, List, Optional
from pathlib import Path

from model_flow.models import ModelFlowObservation, RequestInfo, ModelFlowAction
from .constants import (
    ACTIVE_MODELS,
    COMPLEXITY_MIN_RANK,
    DEMAND_HINT_DELAY,
    HEAT_BUCKET_LOW,
    HEAT_BUCKET_MEDIUM,
    HEAT_FAIL_CAP,
    HEAT_FAIL_SLOPE,
    QUANT_TO_TIER,
    ROLE_TO_MODEL,
    SLA_BASE_STEPS,
    SLA_FLOOR_STEPS,
    SLA_TIGHTEN_BY,
    SLA_TIGHTEN_EVERY,
    TASKS,
    TIER_RANK,
    QUANTS,
)
from .metrics_loader import load_roster
import model_flow.rewards as R


# ---------------------------------------------------------------------------
# Tick caps — bound the internal simulation loops so the reward signal
# remains learnable. The agent can still observe total_time_s to understand
# actual throughput; these caps only limit age-penalty accumulation.
# ---------------------------------------------------------------------------

MAX_EXEC_TICKS = 8    # max internal clock ticks during a single EXECUTE
MAX_LOAD_TICKS = 4    # max internal clock ticks during a single LOAD/REPLACE


# ---------------------------------------------------------------------------
# Stochastic samplers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Heat bucket label
# ---------------------------------------------------------------------------

def _heat_bucket(heat: int) -> str:
    if heat <= HEAT_BUCKET_LOW:
        return "low"
    if heat <= HEAT_BUCKET_MEDIUM:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class ModelFlowEnvironment():
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, benchmark_json: str | None = None):
        super().__init__()

        project_root = Path(__file__).resolve().parents[1]
        if benchmark_json is None:
            benchmark_json = project_root / "Data" / "combined_model_metrics.json"
        else:
            benchmark_json = Path(benchmark_json)
            if not benchmark_json.is_absolute():
                benchmark_json = (project_root / benchmark_json).resolve()

        self.HARDWARE_RAM_MB    = 8000
        self.MAX_STEPS          = 1800
        self.SPIKE_PROB         = 0.10
        self.SPIKE_MB_MIN       = 500
        self.SPIKE_MB_MAX       = 2000
        self.SPIKE_DURATION_MIN = 3
        self.SPIKE_DURATION_MAX = 10
        self.SYSTEM_OVERHEAD_MB = 1100

        self.roster: Dict[str, dict] = load_roster(str(benchmark_json))
        self.evicted_cache: Dict[str, float] = {}

        self.role_to_model: Dict[str, str] = {
            "chatbot":    "gemma-3-4b",
            "translator": "llama_1b",
            "coder":      "qwen3.5-2b",
        }

        self.available_quants: Dict[str, List[str]] = {}
        for model_id in ACTIVE_MODELS:
            self.available_quants[model_id] = sorted(
                {e["quant"] for e in self.roster.values() if e["model"] == model_id}
            )

        self.tasks = TASKS

        # Episode state (initialised in reset)
        self.loaded_models:    Dict[str, dict] = {}
        self.ram_used_mb:      float = 0.0
        self.queue:            List[RequestInfo] = []
        self.completed:        int = 0
        self.step_count:       int = 0
        self.cumulative_reward: float = 0.0
        self.last_feedback:    Optional[str] = None
        self.last_error:       Optional[str] = None
        self.active_spike_mb:  int = 0
        self.spike_steps_remaining: int = 0
        self.failed_execute_history: Dict[str, int] = {}

        self.load_count:          int = 0
        self.evict_count:         int = 0
        self.idle_steps:          int = 0
        self.oom_errors:          int = 0
        self.reasoning_completed: int = 0
        self.quality_failures:    int = 0

        self.current_task:              str = "single-load"
        self._initial_request_count:   int = 0
        self._initial_reasoning_count: int = 0

        self.completion_ages:     List[float] = []
        self.throughput_samples:  List[float] = []
        self.overprovision_count: int = 0
        self.sla_at_serve:        List[int] = []

        # V2 hidden state
        self._model_heat:               Dict[str, int] = {}
        self._recent_quality_outcomes:  List[bool] = []

        self._demand_shift_enabled: bool = False
        self._t_shift:              int  = 999
        self._post_shift:           bool = False
        self._shift_detected_at:    Optional[int] = None

        self._sla_tightening_enabled: bool = False
        self._current_sla_steps:      int  = SLA_BASE_STEPS

        self._deferred:     List[RequestInfo] = []
        self._deferred_ids: set = set()
        self.deferred_served:    int = 0
        self.deferred_abandoned: int = 0

    # -------------------------------------------------------------------------
    # reset
    # -------------------------------------------------------------------------

    def reset(self, task_name: str = "single-load", **kwargs):
        task_cfg = self.tasks.get(task_name, self.tasks["single-load"])

        if task_name in ("quality-limit", "ram-pressure"):
            self.HARDWARE_RAM_MB = 6200
            self.SPIKE_MB_MAX    = 2500
            self.SPIKE_PROB      = 0.18
        else:
            self.HARDWARE_RAM_MB = 8000
            self.SPIKE_MB_MAX    = 2000
            self.SPIKE_PROB      = 0.10

        self.loaded_models          = {}
        self.ram_used_mb            = 0.0
        self.completed              = 0
        self.step_count             = 0
        self.cumulative_reward      = 0.0
        self.last_feedback          = None
        self.last_error             = None
        self.active_spike_mb        = 0
        self.spike_steps_remaining  = 0
        self.failed_execute_history = {}
        self.evicted_cache          = {}

        self.load_count          = 0
        self.evict_count         = 0
        self.idle_steps          = 0
        self.oom_errors          = 0
        self.reasoning_completed = 0
        self.quality_failures    = 0

        self.completion_ages     = []
        self.throughput_samples  = []
        self.overprovision_count = 0
        self.sla_at_serve        = []

        self._deferred           = []
        self._deferred_ids       = set()
        self.deferred_served     = 0
        self.deferred_abandoned  = 0

        self._model_heat              = {}
        self._recent_quality_outcomes = []

        self._demand_shift_enabled = task_cfg.get("demand_shift", False)
        if self._demand_shift_enabled:
            self._t_shift = random.randint(5, 8)
        else:
            self._t_shift = 999
        self._post_shift        = False
        self._shift_detected_at = None

        self._sla_tightening_enabled = task_cfg.get("sla_tightening", False)
        self._current_sla_steps      = SLA_BASE_STEPS

        req_list = task_cfg["requests"]
        self.queue = []
        for i, req in enumerate(req_list):
            r = RequestInfo(request_id=f"u_{i}", **req)
            r.prompt_tokens = 128 if r.complexity == "reasoning" else 64
            r.gen_tokens    = 512 if r.complexity == "reasoning" else 128
            self.queue.append(r)

        self.current_task             = task_name
        self._initial_request_count   = len(self.queue)
        self._initial_reasoning_count = sum(1 for r in self.queue if r.reasoning)

        return self._get_observation()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _tick_spike(self):
        if self.spike_steps_remaining > 0:
            self.spike_steps_remaining -= 1
            if self.spike_steps_remaining == 0:
                self.active_spike_mb = 0

        if self.active_spike_mb == 0 and random.random() < self.SPIKE_PROB:
            self.active_spike_mb       = random.randint(self.SPIKE_MB_MIN, self.SPIKE_MB_MAX)
            self.spike_steps_remaining = random.randint(
                self.SPIKE_DURATION_MIN, self.SPIKE_DURATION_MAX
            )

    def _tick_sla(self):
        if not self._sla_tightening_enabled:
            return
        if self.step_count > 0 and self.step_count % SLA_TIGHTEN_EVERY == 0:
            self._current_sla_steps = max(
                SLA_FLOOR_STEPS,
                self._current_sla_steps - SLA_TIGHTEN_BY,
            )

    def _tick_demand_shift(self):
        if not self._demand_shift_enabled or self._post_shift:
            return
        if self.step_count < self._t_shift:
            return

        self._post_shift        = True
        self._shift_detected_at = self.step_count + DEMAND_HINT_DELAY

        type_counts: Dict[str, int] = {}
        for req in self.queue:
            type_counts[req.model_type] = type_counts.get(req.model_type, 0) + 1
        if not type_counts:
            return

        dominant = max(type_counts, key=type_counts.get)
        flip_map = {
            "chatbot":    "coder",
            "coder":      "translator",
            "translator": "chatbot",
        }
        new_type = flip_map.get(dominant, "coder")

        inject = [
            RequestInfo(
                request_id=f"shift_{i}",
                model_type=new_type,
                complexity="reasoning" if i == 1 else "standard",
                prompt_tokens=128 if i == 1 else 64,
                gen_tokens=512 if i == 1 else 128,
            )
            for i in range(3)
        ]
        self.queue.extend(inject)

    def _is_model_needed(self, model_id: str) -> bool:
        for req in self.queue:
            if self.role_to_model.get(req.model_type) == model_id:
                return True
        for req in self._deferred:
            if self.role_to_model.get(req.model_type) == model_id:
                return True
        return False

    def _reinject_deferred(self):
        if self._deferred:
            self.queue = self._deferred + self.queue
            self._deferred = []

    # -------------------------------------------------------------------------
    # step
    # -------------------------------------------------------------------------

    def step(self, action: ModelFlowAction):
        self.last_error    = None
        self.last_feedback = None
        done   = False
        reward = 0.0

        def _clock_tick():
            nonlocal reward
            self.step_count += 1
            self._tick_spike()
            self._tick_sla()
            self._tick_demand_shift()
            for req in self.queue:
                req.age_steps += 1
            for req in self._deferred:
                req.age_steps += 1
            reward += R.clock_tick_penalty(self.queue, self.loaded_models)

        # LOAD 
        if action.command == "LOAD":
            _clock_tick()
            if not action.model_id or not action.quant_type:
                self.last_error = "LOAD requires model_id and quant_type"
            else:
                key = f"{action.model_id}-{action.quant_type}"
                if key not in self.roster:
                    self.last_error = f"Unknown config: {key}"
                elif key in self.loaded_models:
                    self.last_error = f"{key} is already loaded"
                    reward += R.load_already_loaded()
                else:
                    data      = self.roster[key]
                    host_size = _sample_host_mb(data)
                    eff_free  = (
                        self.HARDWARE_RAM_MB
                        - self.ram_used_mb
                        - self.active_spike_mb
                        - self.SYSTEM_OVERHEAD_MB
                    )

                    if host_size > eff_free:
                        self.last_error = (
                            f"OOM: {key} needs {host_size:.0f}MB, "
                            f"only {eff_free:.0f}MB free"
                        )
                        reward += R.load_oom()
                        self.oom_errors += 1
                    else:
                        is_warm = key in self.evicted_cache
                        if is_warm:
                            actual_load_s = data["load_avg_ms"] / 1000.0 * random.uniform(0.1, 0.2)
                            self.evicted_cache.pop(key)
                        else:
                            actual_load_s = _sample_load_time_s(data)

                        # TICK CAP for load time 
                        # Compute raw ticks but cap at MAX_LOAD_TICKS.
                        raw_load_ticks = max(1, math.ceil(actual_load_s))
                        load_ticks     = min(raw_load_ticks, MAX_LOAD_TICKS)
                        for _ in range(load_ticks - 1):
                            _clock_tick()

                        self._model_heat[key] = self._model_heat.get(key, 0) + 1

                        self.loaded_models[key] = {
                            "model":      action.model_id,
                            "quant":      action.quant_type,
                            "tier":       data["tier"],
                            "size_mb":    round(host_size),
                            "weight_mb":  round(data["size_mb"]),
                            "cpu_avg":    data["cpu_avg"],
                            "gen_tps":    data["gen_tps"],
                            "prompt_tps": data["prompt_tps"],
                            "ctx_mb":     data["ctx_mb"],
                            "comp_mb":    data["comp_mb"],
                            "kv_mb_max":  data["kv_mb_max"],
                        }
                        self.ram_used_mb += host_size
                        self.load_count  += 1

                        reward += R.load_success(actual_load_s, len(self.loaded_models))
                        warm_str = " (warm)" if is_warm else " (cold)"
                        self.last_feedback = (
                            f"Loaded {key}{warm_str} in {actual_load_s:.1f}s"
                            f" (ticks_used={load_ticks}/{raw_load_ticks})."
                        )

        # EXECUTE
        elif action.command == "EXECUTE":
            _clock_tick()
            if not action.model_id or not action.quant_type:
                self.last_error = "EXECUTE requires model_id and quant_type"
                reward += R.execute_bad_args()
            else:
                key = f"{action.model_id}-{action.quant_type}"
                if key not in self.loaded_models:
                    self.last_error = f"{key} is not loaded"
                    reward += R.execute_not_loaded()
                elif not self.queue and not self._deferred:
                    self.last_error = "Queue is empty"
                    reward += R.execute_empty_queue()
                else:
                    self._reinject_deferred()

                    slot         = self.loaded_models[key]
                    target_model = action.model_id
                    matching     = []

                    for req in self.queue:
                        req_model = self.role_to_model.get(req.model_type, "")
                        if (
                            req_model == target_model
                            and TIER_RANK[slot["tier"]] >= COMPLEXITY_MIN_RANK[req.complexity]
                        ):
                            matching.append(req)
                        if len(matching) >= min(action.batch_size or 8, 8):
                            break

                    if not matching:
                        self.failed_execute_history[key] = (
                            self.failed_execute_history.get(key, 0) + 1
                        )
                        fails = self.failed_execute_history[key]
                        tier_mismatch = any(
                            self.role_to_model.get(req.model_type) == target_model
                            and TIER_RANK[slot["tier"]] < COMPLEXITY_MIN_RANK[req.complexity]
                            for req in self.queue
                        )
                        if tier_mismatch:
                            self.last_error = (
                                f"TIER TOO LOW: {key} cannot serve reasoning requests. "
                                f"Need Q6_K or higher."
                            )
                        else:
                            self.last_error = f"No matching requests for {key} (fail #{fails})"
                        reward += R.execute_no_match(fails)
                    else:
                        batch = len(matching)
                        max_needed_rank = max(
                            COMPLEXITY_MIN_RANK[r.complexity] for r in matching
                        )
                        if TIER_RANK[slot["tier"]] > max_needed_rank:
                            self.overprovision_count += 1

                        kv_dynamic   = slot["kv_mb_max"] * (batch / 8.0)
                        peak_dynamic = slot["ctx_mb"] + slot["comp_mb"] + kv_dynamic
                        total_peak   = (
                            self.ram_used_mb
                            + self.active_spike_mb
                            + self.SYSTEM_OVERHEAD_MB
                            + peak_dynamic
                        )

                        if total_peak > self.HARDWARE_RAM_MB:
                            self.last_error = f"RUNTIME OOM: peaked at {total_peak:.0f}MB"
                            reward += R.execute_runtime_oom()
                        else:
                            heat       = self._model_heat.get(key, 0)
                            quality_ok = _quality_failure(heat)

                            if not quality_ok:
                                self.quality_failures += 1
                                reward += R.quality_degraded_penalty()

                            self._recent_quality_outcomes.append(quality_ok)
                            if len(self._recent_quality_outcomes) > 3:
                                self._recent_quality_outcomes.pop(0)

                            tier_multipliers = {
                                "low": 1.0, "medium": 1.1, "high": 1.2, "risky": 1.2,
                            }

                            reward += R.execute_success(
                                matching_requests=matching,
                                tier_multipliers=tier_multipliers,
                                slot_tier=slot["tier"],
                                quant_type=action.quant_type,
                                roster_data=self.roster[key],
                                quality_ok=quality_ok,
                                current_sla_steps=self._current_sla_steps,
                            )

                            processed_ids = set()
                            self.failed_execute_history[key] = 0

                            for req in matching:
                                self.completed += 1
                                if req.reasoning:
                                    self.reasoning_completed += 1
                                processed_ids.add(req.request_id)
                                self.completion_ages.append(req.age_steps)
                                self.sla_at_serve.append(self._current_sla_steps)

                                if req.request_id in self._deferred_ids:
                                    reward += R.defer_serve_bonus(req.age_steps, quality_ok)
                                    self._deferred_ids.discard(req.request_id)
                                    self.deferred_served += 1

                            self.queue = [
                                r for r in self.queue
                                if r.request_id not in processed_ids
                            ]

                            sum_cpu     = sum(m["cpu_avg"] for m in self.loaded_models.values())
                            contention  = min(0.8, sum_cpu / 400.0)
                            eff_gen_tps = slot["gen_tps"] * (1.0 - contention)
                            self.throughput_samples.append(eff_gen_tps)

                            p_tok        = max(r.prompt_tokens for r in matching)
                            g_tok        = sum(r.gen_tokens for r in matching) / math.sqrt(batch)
                            total_time_s = (p_tok / slot["prompt_tps"]) + (g_tok / eff_gen_tps)

                            # TICK CAP for exec time
                            # Compute raw ticks but cap at MAX_EXEC_TICKS.
                            # total_time_s is still accurate and reported in
                            # feedback — only the age-penalty ticks are capped.
                            raw_exec_ticks  = max(1, math.ceil(total_time_s))
                            exec_ticks_used = min(raw_exec_ticks, MAX_EXEC_TICKS)
                            for _ in range(exec_ticks_used - 1):
                                _clock_tick()

                            self.last_feedback = (
                                f"Executed batch={batch} on {key} in {total_time_s:.1f}s"
                                f" (ticks={exec_ticks_used}/{raw_exec_ticks},"
                                f" heat={heat}, quality_ok={quality_ok},"
                                f" contention={contention:.1%})."
                            )

        # EVICT
        elif action.command == "EVICT":
            _clock_tick()
            if action.model_id and action.quant_type:
                key = f"{action.model_id}-{action.quant_type}"
            elif self.loaded_models:
                key = list(self.loaded_models.keys())[-1]
            else:
                key = None

            if key and key in self.loaded_models:
                data = self.loaded_models.pop(key)
                self.ram_used_mb -= data.get("size_mb", 0)
                self.evicted_cache[key] = self.step_count
                self.evict_count += 1

                model_name   = key.rsplit("-", 1)[0]
                still_needed = self._is_model_needed(model_name)
                reward += R.evict_success(data.get("size_mb", 0), still_needed)
                self.last_feedback = f"Evicted {key}. Freed {data.get('size_mb', 0):.0f}MB."
            else:
                self.last_error = f"Cannot evict {key or 'nothing'}"
                reward += R.evict_nothing_to_evict()

        # IDLE
        elif action.command == "IDLE":
            _clock_tick()
            reward += R.idle_penalty()
            if self.queue or self._deferred:
                self.idle_steps += 1
            self.last_feedback = "Idled."

        # REPLACE
        elif action.command == "REPLACE":
            evict_key = None
            if action.evict_model_id and action.evict_quant_type:
                evict_key = f"{action.evict_model_id}-{action.evict_quant_type}"
            elif self.loaded_models:
                evict_key = list(self.loaded_models.keys())[0]

            if evict_key and evict_key in self.loaded_models:
                before_count = len(self.loaded_models)
                data = self.loaded_models.pop(evict_key)
                self.ram_used_mb -= data.get("size_mb", 0)
                self.evicted_cache[evict_key] = self.step_count
                self.evict_count += 1

                evicted_model_name = evict_key.rsplit("-", 1)[0]
                evicted_needed     = self._is_model_needed(evicted_model_name)
                reward += R.replace_evict_component(evicted_needed)

                _clock_tick()

                if not action.model_id or not action.quant_type:
                    self.last_error = "REPLACE requires model_id and quant_type"
                    reward += R.replace_bad_load_args()
                else:
                    new_key = f"{action.model_id}-{action.quant_type}"
                    if new_key not in self.roster:
                        self.last_error = f"Unknown config: {new_key}"
                        reward += R.replace_load_unknown_config()
                    elif new_key in self.loaded_models:
                        self.last_error = f"{new_key} is already loaded"
                        reward += R.replace_load_already_loaded()
                    else:
                        new_data  = self.roster[new_key]
                        host_size = _sample_host_mb(new_data)
                        eff_free  = (
                            self.HARDWARE_RAM_MB
                            - self.ram_used_mb
                            - self.active_spike_mb
                            - self.SYSTEM_OVERHEAD_MB
                        )
                        if host_size > eff_free:
                            self.last_error = f"OOM: {new_key} needs {host_size:.0f}MB"
                            reward += R.replace_load_oom()
                            self.oom_errors += 1
                        else:
                            is_warm = new_key in self.evicted_cache
                            if is_warm:
                                actual_load_s = new_data["load_avg_ms"] / 1000.0 * random.uniform(0.1, 0.2)
                                self.evicted_cache.pop(new_key)
                            else:
                                actual_load_s = _sample_load_time_s(new_data)

                            # TICK CAP for replace load time
                            raw_load_ticks = max(1, math.ceil(actual_load_s))
                            load_ticks     = min(raw_load_ticks, MAX_LOAD_TICKS)
                            for _ in range(load_ticks - 1):
                                _clock_tick()

                            self._model_heat[new_key] = self._model_heat.get(new_key, 0) + 1

                            self.loaded_models[new_key] = {
                                "model":      action.model_id,
                                "quant":      action.quant_type,
                                "tier":       new_data["tier"],
                                "size_mb":    round(host_size),
                                "weight_mb":  round(new_data["size_mb"]),
                                "cpu_avg":    new_data["cpu_avg"],
                                "gen_tps":    new_data["gen_tps"],
                                "prompt_tps": new_data["prompt_tps"],
                                "ctx_mb":     new_data["ctx_mb"],
                                "comp_mb":    new_data["comp_mb"],
                                "kv_mb_max":  new_data["kv_mb_max"],
                            }
                            self.ram_used_mb += host_size
                            self.load_count  += 1
                            after_count = len(self.loaded_models)

                            reward += R.replace_load_success(
                                actual_load_s, before_count, after_count
                            )
                            self.last_feedback = (
                                f"REPLACE: evicted {evict_key},"
                                f" loaded {new_key} in {actual_load_s:.1f}s"
                                f" (ticks={load_ticks}/{raw_load_ticks})."
                            )
            else:
                self.last_error = f"Cannot replace {evict_key or 'nothing'}"
                reward += R.replace_no_target()

        # DEFER
        elif action.command == "DEFER":
            _clock_tick()
            if not self.queue:
                self.last_error = "Nothing to defer — queue is empty"
                reward += R.execute_empty_queue()
            else:
                target_type  = action.model_id
                deferred_req = None
                if target_type:
                    for req in self.queue:
                        if req.model_type == target_type:
                            deferred_req = req
                            break
                if deferred_req is None:
                    deferred_req = self.queue[0]

                self.queue.remove(deferred_req)
                deferred_req.age_steps += 3
                self._deferred.append(deferred_req)
                self._deferred_ids.add(deferred_req.request_id)

                reward += R.defer_penalty(deferred_req.age_steps)
                self.last_feedback = (
                    f"Deferred {deferred_req.request_id}"
                    f" ({deferred_req.model_type}/{deferred_req.complexity})."
                    f" Deferred queue: {len(self._deferred)}."
                )

        # Episode terminal conditions
        all_done = not self.queue and not self._deferred
        if all_done:
            done = True
            reward += R.episode_success()
            self.last_feedback = "SUCCESS: all requests served!"
        elif self.step_count >= self.MAX_STEPS:
            done = True
            self.deferred_abandoned += len(self._deferred)
            reward += R.episode_timeout()

        if self.last_error:
            self.last_feedback = f"ERROR: {self.last_error}"

        self.cumulative_reward += reward
        obs        = self._get_observation()
        obs.done   = done
        obs.reward = reward
        return obs


    # _get_observation
    def _get_observation(self) -> ModelFlowObservation:
        model_summary = {}
        for role, model_id in self.role_to_model.items():
            stats = {}
            for quant in self.available_quants.get(model_id, []):
                key = f"{model_id}-{quant}"
                if key in self.roster:
                    d = self.roster[key]
                    stats[quant] = {
                        "size_mb":    round(d["host_mb"]),
                        "gen_tps":    d["gen_tps"],
                        "prompt_tps": d["prompt_tps"],
                        "cpu_avg":    d["cpu_avg"],
                        "tier":       d["tier"],
                    }
            model_summary[role] = {"model_id": model_id, "stats": stats}

        heat_signals = {
            key: _heat_bucket(self._model_heat.get(key, 0))
            for key in self.loaded_models
        }

        demand_hint: Optional[str] = None
        if (
            self._post_shift
            and self._shift_detected_at is not None
            and self.step_count >= self._shift_detected_at
        ):
            demand_hint = "shift_detected"

        return ModelFlowObservation(
            ram_used_mb=int(self.ram_used_mb),
            ram_limit_mb=self.HARDWARE_RAM_MB,
            loaded_models={k: v for k, v in self.loaded_models.items()},
            queue=self.queue.copy(),
            model_summary=model_summary,
            last_action_feedback=self.last_feedback,
            step_count=self.step_count,
            pressure_spike_mb=self.active_spike_mb,
            spike_steps_remaining=self.spike_steps_remaining,
            last_action_error=self.last_error,
            model_heat_signals=heat_signals,
            recent_quality_outcomes=self._recent_quality_outcomes.copy(),
            demand_hint=demand_hint,
            current_sla_steps=self._current_sla_steps,
            deferred_count=len(self._deferred),
            info={
                "cumulative_reward": round(self.cumulative_reward, 2),
                "completed":         self.completed,
                "pending":           len(self.queue),
                "deferred":          len(self._deferred),
                "ram_free_mb":       round(
                    self.HARDWARE_RAM_MB - self.ram_used_mb - self.active_spike_mb
                ),
                "current_sla_steps": self._current_sla_steps,
                "grader_metrics": {
                    "loads":              self.load_count,
                    "evicts":             self.evict_count,
                    "idles":              self.idle_steps,
                    "ooms":               self.oom_errors,
                    "reasoning_done":     self.reasoning_completed,
                    "quality_failures":   self.quality_failures,
                    "deferred_served":    self.deferred_served,
                    "deferred_abandoned": self.deferred_abandoned,
                },
            },
            reward=0.0,
        )

    # state / scoring

    def state(self):
        return {
            "step":                  self.step_count,
            "ram_used_mb":           self.ram_used_mb,
            "pending":               len(self.queue),
            "deferred":              len(self._deferred),
            "loaded_models":         list(self.loaded_models.keys()),
            "active_spike_mb":       self.active_spike_mb,
            "spike_steps_remaining": self.spike_steps_remaining,
            "current_sla_steps":     self._current_sla_steps,
        }

    def get_episode_result(self):
        from graders import EpisodeResult
        return EpisodeResult(
            task_name=self.current_task,
            total_requests=self._initial_request_count,
            completed_requests=self.completed,
            total_reasoning=self._initial_reasoning_count,
            reasoning_completed=self.reasoning_completed,
            steps_taken=self.step_count,
            max_steps=self.MAX_STEPS,
            load_count=self.load_count,
            evict_count=self.evict_count,
            oom_errors=self.oom_errors,
            idle_steps=self.idle_steps,
            completion_ages=self.completion_ages.copy(),
            throughput_samples=self.throughput_samples.copy(),
            overprovision_count=self.overprovision_count,
            quality_failures=self.quality_failures,
            deferred_served=self.deferred_served,
            deferred_abandoned=self.deferred_abandoned,
            sla_at_serve=self.sla_at_serve.copy(),
        )

    def score_task(self) -> float:
        from graders import grade
        result = self.get_episode_result()
        return grade(self.current_task, result)