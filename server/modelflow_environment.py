import os
import math
import random
import json
import sys
from typing import List, Dict

from groq import Groq
from openenv.core.env_server import Environment
from models import ModelFlowObservation, RequestInfo, ModelFlowAction

# ── Constants ────────────────────────────────────────────────────────────────

ACTIVE_MODELS = {"qwen3.5-2b", "llama_1b", "gemma-3-4b"}
QUANTS = {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}

# Quant name → tier label
QUANT_TO_TIER: Dict[str, str] = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K": "high",
    "Q8_0": "risky",
}

# Numeric rank used for tier comparisons
TIER_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "risky": 2}

# Minimum tier rank required to serve each complexity level
COMPLEXITY_MIN_RANK: Dict[str, int] = {"standard": 0, "reasoning": 2}


class ModelFlowEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Groq helper ──────────────────────────────────────────────────────────

    def _init_groq_client(self):
        api_key = os.getenv("GROQ_API_KEY")
        return Groq(api_key=api_key) if api_key else None

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        benchmark_json: str = "model_flow/Data/combined_model_metrics.json",
    ):
        super().__init__()

        # Robust data path check (handles local vs. container structure)
        if not os.path.exists(benchmark_json):
            alt_path = benchmark_json.replace("model_flow/", "")
            if os.path.exists(alt_path):
                benchmark_json = alt_path

        self.HARDWARE_RAM_MB = 8000
        self.MAX_STEPS = 1800
        self.LOAD_NOISE_SIG = 0.05
        self.SPIKE_PROB = 0.1
        self.SPIKE_MB_MIN = 500
        self.SPIKE_MB_MAX = 2000
        self.SPIKE_DURATION_MIN = 3
        self.SPIKE_DURATION_MAX = 10
        self.SYSTEM_OVERHEAD_MB = 1100

        self.roster: Dict[str, dict] = {}
        self.evicted_cache: Dict[str, float] = {}

        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "combined_model_metrics.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            for m in data["models"]:
                parts = m["model_key"].rsplit("-", 1)
                model_name = parts[0]
                quant_type = parts[1]

                if model_name in ACTIVE_MODELS and quant_type in QUANTS:
                    self.roster[m["model_key"]] = {
                        "model": model_name,
                        "quant": quant_type,
                        "tier": QUANT_TO_TIER[quant_type],
                        "size_mb": m["weight_size_mb"],
                        "host_mb": m["memory"]["host_total_mb_avg"],
                        "ctx_mb": m["memory"]["context_mb_avg"],
                        "comp_mb": m["memory"]["compute_mb_avg"],
                        "kv_mb_max": m["memory"]["kv_total_mb_max"],
                        "prompt_tps": m["speed"]["prompt_tps"],
                        "gen_tps": m["speed"]["gen_tps"],
                        "load_avg_ms": m["speed"]["load_time_avg_ms"],
                        "cpu_avg": m["cpu"]["avg_cpu_percent"],
                    }
        else:
            self.roster = {
                "gemma-3-4b-Q4_K_M": {
                    "model": "gemma-3-4b",
                    "quant": "Q4_K_M",
                    "tier": "low",
                    "host_mb": 2583.0,
                    "size_mb": 2300,
                    "ctx_mb": 40.0,
                    "comp_mb": 110.0,
                    "kv_mb_max": 96.0,
                    "prompt_tps": 120.0,
                    "gen_tps": 24.0,
                    "load_avg_ms": 1000.0,
                    "cpu_avg": 30.0,
                },
            }

        self.role_to_model: Dict[str, str] = {
            "chatbot": "gemma-3-4b",
            "translator": "llama_1b",
            "coder": "qwen3.5-2b",
        }

        self.available_quants: Dict[str, List[str]] = {}
        for model_id in ACTIVE_MODELS:
            self.available_quants[model_id] = sorted(
                {e["quant"] for e in self.roster.values() if e["model"] == model_id}
            )

        self.tasks = {
            "single-load": {"requests": [{"model_type": "chatbot", "complexity": "standard"}] * 9},
            "multi-load": {"requests": [
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "standard"},
                {"model_type": "translator", "complexity": "standard"},
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "standard"},
                {"model_type": "translator", "complexity": "reasoning"},
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "standard"},
                {"model_type": "translator", "complexity": "reasoning"},
                {"model_type": "coder", "complexity": "reasoning"},
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "reasoning"}
            ]},
            "quality-limit": {"requests": [
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "translator", "complexity": "standard"},
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "standard"},
                {"model_type": "translator", "complexity": "standard"},
                {"model_type": "chatbot", "complexity": "reasoning"},
                {"model_type": "coder", "complexity": "standard"},
                {"model_type": "translator", "complexity": "reasoning"},
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "reasoning"},
                {"model_type": "chatbot", "complexity": "standard"},
                {"model_type": "coder", "complexity": "standard"},
                {"model_type": "translator", "complexity": "standard"},
                {"model_type": "coder", "complexity": "reasoning"}
            ]},
            "ram-pressure": {
                "requests": [
                    {"model_type": "coder", "complexity": "standard"},
                    {"model_type": "chatbot", "complexity": "standard"},
                    {"model_type": "translator", "complexity": "standard"},
                    {"model_type": "coder", "complexity": "reasoning"},
                    {"model_type": "chatbot", "complexity": "reasoning"},
                    {"model_type": "translator", "complexity": "reasoning"},
                    {"model_type": "coder", "complexity": "reasoning"},
                    {"model_type": "chatbot", "complexity": "standard"},
                    {"model_type": "translator", "complexity": "reasoning"},
                    {"model_type": "coder", "complexity": "standard"},
                    {"model_type": "chatbot", "complexity": "reasoning"},
                    {"model_type": "translator", "complexity": "standard"}
                ]
            }
        }

        self.loaded_models: Dict[str, dict] = {}
        self.ram_used_mb: float = 0.0
        self.queue: List[RequestInfo] = []
        self.completed: int = 0
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.last_feedback: str | None = None
        self.last_error: str | None = None
        self.active_spike_mb: int = 0
        self.spike_steps_remaining: int = 0
        self.failed_execute_history: Dict[str, int] = {}

        self.load_count: int = 0
        self.evict_count: int = 0
        self.idle_steps: int = 0
        self.oom_errors: int = 0
        self.reasoning_completed: int = 0

        self.current_task: str = "single-load"
        self._initial_request_count: int = 0
        self._initial_reasoning_count: int = 0

        self.groq_client = self._init_groq_client()
        self.groq_model = "llama3-8b-8192"

        self.completion_ages: List[float] = []
        self.throughput_samples: List[float] = []
        self.overprovision_count = 0

    def reset(self, task_name: str = "single-load", **kwargs):
        if task_name in ["quality-limit", "ram-pressure"]:
            self.HARDWARE_RAM_MB = 6200
            self.SPIKE_MB_MAX = 2500
            self.SPIKE_PROB = 0.18
        else:
            self.HARDWARE_RAM_MB = 8000
            self.SPIKE_MB_MAX = 2000
            self.SPIKE_PROB = 0.10

        self.loaded_models = {}
        self.ram_used_mb = 0.0
        req_list = self.tasks.get(task_name, self.tasks["single-load"])["requests"]
        self.queue = []
        for i, req in enumerate(req_list):
            r = RequestInfo(request_id=f"u_{i}", **req)
            if r.complexity == "reasoning":
                r.prompt_tokens = 128
                r.gen_tokens = 512
            else:
                r.prompt_tokens = 64
                r.gen_tokens = 128
            self.queue.append(r)

        self.completed = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.last_feedback = None
        self.last_error = None
        self.active_spike_mb = 0
        self.spike_steps_remaining = 0
        self.failed_execute_history = {}

        self.current_task = task_name
        self._initial_request_count = len(self.queue)
        self._initial_reasoning_count = sum(1 for r in self.queue if r.reasoning)

        self.load_count = 0
        self.evict_count = 0
        self.idle_steps = 0
        self.oom_errors = 0
        self.reasoning_completed = 0

        self.completion_ages = []
        self.throughput_samples = []
        self.overprovision_count = 0
        
        # print(f"[GRADER] Target Task: {task_name} | RAM Limit: {self.HARDWARE_RAM_MB} MB", file=sys.stderr)
        # print(f"\n[GRADER] Target Task: {self.current_task} | Requests: {self._initial_request_count} (Reasoning: {self._initial_reasoning_count})", file=sys.stderr)
        return self._get_observation()

    def _tick_spike(self):
        if self.spike_steps_remaining > 0:
            self.spike_steps_remaining -= 1
            if self.spike_steps_remaining == 0:
                self.active_spike_mb = 0

        if self.active_spike_mb == 0 and random.random() < self.SPIKE_PROB:
            self.active_spike_mb = random.randint(self.SPIKE_MB_MIN, self.SPIKE_MB_MAX)
            self.spike_steps_remaining = random.randint(self.SPIKE_DURATION_MIN, self.SPIKE_DURATION_MAX)

    def _is_model_needed(self, model_id: str) -> bool:
        for req in self.queue:
            if self.role_to_model.get(req.model_type) == model_id:
                return True
        return False

    def step(self, action: ModelFlowAction):
        self.last_error = None
        self.last_feedback = None
        done = False
        reward = 0.0

        def _clock_tick():
            nonlocal reward
            self.step_count += 1
            self._tick_spike()
            for req in self.queue:
                req.age_steps += 1
                penalty_val = 0.005 * (req.age_steps ** 1.05)
                reward -= min(penalty_val, 0.5)

            if len(self.queue) > 0 and len(self.loaded_models) == 0:
                reward -= 2.0

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
                    reward -= 5.0
                else:
                    data = self.roster[key]
                    host_size = data["host_mb"]
                    effective_free = self.HARDWARE_RAM_MB - self.ram_used_mb - self.active_spike_mb - self.SYSTEM_OVERHEAD_MB

                    if host_size > effective_free:
                        self.last_error = f"OOM: {key} needs {host_size:.0f}MB, only {effective_free:.0f}MB free"
                        reward -= 30.0
                        self.oom_errors += 1
                        # print(f"[GRADER] OOM Error recorded! Total: {self.oom_errors}", file=sys.stderr)
                    else:
                        is_warm = key in self.evicted_cache
                        base_load_s = data["load_avg_ms"] / 1000.0
                        if is_warm:
                            actual_load_s = base_load_s * random.uniform(0.1, 0.2)
                            self.evicted_cache.pop(key)
                        else:
                            sigma = 0.15
                            mu = math.log(max(base_load_s, 1e-6))
                            actual_load_s = random.lognormvariate(mu, sigma)

                        load_steps = max(1, math.ceil(actual_load_s))
                        for _ in range(load_steps - 1):
                            _clock_tick()

                        self.loaded_models[key] = {
                            "model": action.model_id,
                            "quant": action.quant_type,
                            "tier": data["tier"],
                            "size_mb": round(host_size),
                            "weight_mb": round(data["size_mb"]),
                            "cpu_avg": data["cpu_avg"],
                            "gen_tps": data["gen_tps"],
                            "prompt_tps": data["prompt_tps"],
                            "ctx_mb": data["ctx_mb"],
                            "comp_mb": data["comp_mb"],
                            "kv_mb_max": data["kv_mb_max"],
                        }
                        self.ram_used_mb += host_size
                        self.load_count += 1
                        # print(f"[GRADER] LOAD action: {key} | Total Loads: {self.load_count}", file=sys.stderr)
                        reward -= actual_load_s * 1.5

                        if len(self.loaded_models) >= 2:
                            reward += 3.0
                            # print(f"[GRADER] Multi-model Bonus: {len(self.loaded_models)} models co-resident.", file=sys.stderr)

                        warm_str = " (warm)" if is_warm else " (cold)"
                        self.last_feedback = f"Loaded {key}{warm_str} in {actual_load_s:.1f}s."

        elif action.command == "EXECUTE":
            _clock_tick()
            if not action.model_id or not action.quant_type:
                self.last_error = "EXECUTE requires model_id and quant_type"
                reward -= 5.0
            else:
                key = f"{action.model_id}-{action.quant_type}"
                if key not in self.loaded_models:
                    self.last_error = f"{key} is not loaded"
                    reward -= 10.0
                elif not self.queue:
                    self.last_error = "Queue is empty"
                    reward -= 5.0
                else:
                    slot = self.loaded_models[key]
                    target_model = action.model_id
                    matching = []

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
                        self.failed_execute_history[key] = self.failed_execute_history.get(key, 0) + 1
                        fails = self.failed_execute_history[key]
                        penalty = 5.0 * (fails ** 2)

                        tier_mismatch = any(
                            self.role_to_model.get(req.model_type) == target_model
                            and TIER_RANK[slot["tier"]] < COMPLEXITY_MIN_RANK[req.complexity]
                            for req in self.queue
                        )

                        if tier_mismatch:
                            self.last_error = (
                                f"TIER TOO LOW: {key} (tier={slot['tier']}) cannot serve reasoning requests. "
                                f"Need Q6_K or higher. Try REPLACE with higher quant."
                            )
                        else:
                            self.last_error = f"No matching requests for {key} (fail #{fails})"

                        reward -= penalty
                    else:
                        batch = len(matching)
                        max_needed_rank = max(COMPLEXITY_MIN_RANK[r.complexity] for r in matching)
                        if TIER_RANK[slot["tier"]] > max_needed_rank:
                            self.overprovision_count += 1
                        kv_dynamic = slot["kv_mb_max"] * (batch / 8.0)
                        peak_dynamic = slot["ctx_mb"] + slot["comp_mb"] + kv_dynamic
                        total_peak = self.ram_used_mb + self.active_spike_mb + self.SYSTEM_OVERHEAD_MB + peak_dynamic

                        if total_peak > self.HARDWARE_RAM_MB:
                            self.last_error = f"RUNTIME OOM: {key} peaked at {total_peak:.0f}MB"
                            reward -= 50.0
                        else:
                            processed_ids = []
                            self.failed_execute_history[key] = 0
                            tier_multipliers = {"low": 1.0, "medium": 1.1, "high": 1.2, "risky": 1.2}
                            multiplier = tier_multipliers.get(slot["tier"], 1.0)

                            for req in matching:
                                gain = 25.0 if req.complexity == "reasoning" else 15.0
                                reward += (gain * multiplier)
                                self.completed += 1
                                if req.reasoning:
                                    self.reasoning_completed += 1
                                    # print(f"[GRADER] Reasoning request completed!", file=sys.stderr)
                                processed_ids.append(req.request_id)
                                self.completion_ages.append(req.age_steps)

                            self.queue = [r for r in self.queue if r.request_id not in processed_ids]

                            sum_cpu = sum(m["cpu_avg"] for m in self.loaded_models.values())
                            contention = min(0.8, sum_cpu / 400.0)
                            eff_gen_tps = slot["gen_tps"] * (1.0 - contention)
                            self.throughput_samples.append(eff_gen_tps)
                            p_tok = max(r.prompt_tokens for r in matching)
                            g_tok = sum(r.gen_tokens for r in matching) / math.sqrt(batch)
                            total_time_s = (p_tok / slot["prompt_tps"]) + (g_tok / eff_gen_tps)

                            exec_steps = max(1, math.ceil(total_time_s))
                            for _ in range(exec_steps - 1):
                                _clock_tick()

                            self.last_feedback = f"Executed batch {batch} on {key} in {total_time_s:.1f}s (contention: {contention:.1%})."

                # self.last_feedback = f"Remote executed {processed} requests via Groq."

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
                # print(f"[GRADER] EVICT action: {key} | Total Evicts: {self.evict_count}", file=sys.stderr)
                reward -= 10.0

                model_name = key.rsplit("-", 1)[0]
                if not self._is_model_needed(model_name):
                    reward += 5.0
                    # print(f"[GRADER] Clean Eviction Bonus: {model_name} is no longer needed.", file=sys.stderr)

                self.last_feedback = f"Evicted {key}. Freed {data.get('size_mb', 0)}MB."
            else:
                self.last_error = f"Cannot evict {key or 'nothing'}"
                reward -= 5.0

        elif action.command == "IDLE":
            _clock_tick()
            reward -= 15.0
            if self.queue:
                self.idle_steps += 1
                # print(f"[GRADER] IDLE waste recorded. Total: {self.idle_steps}", file=sys.stderr)
            self.last_feedback = "Idled."

        elif action.command == "REPLACE":
            evict_key = None
            if action.evict_model_id and action.evict_quant_type:
                evict_key = f"{action.evict_model_id}-{action.evict_quant_type}"
            elif self.loaded_models:
                evict_key = list(self.loaded_models.keys())[0]

            if evict_key and evict_key in self.loaded_models:
                before_loaded_count = len(self.loaded_models)

                data = self.loaded_models.pop(evict_key)
                self.ram_used_mb -= data.get("size_mb", 0)
                self.evicted_cache[evict_key] = self.step_count
                self.evict_count += 1

                evicted_model_name = evict_key.rsplit("-", 1)[0]
                evicted_needed = self._is_model_needed(evicted_model_name)

                if not evicted_needed:
                    reward += 2.0
                else:
                    reward -= 15.0

                _clock_tick()

                if not action.model_id or not action.quant_type:
                    self.last_error = "REPLACE requires model_id and quant_type"
                    reward -= 5.0
                else:
                    new_key = f"{action.model_id}-{action.quant_type}"

                    if new_key not in self.roster:
                        self.last_error = f"Unknown config: {new_key}"
                        reward -= 5.0

                    elif new_key in self.loaded_models:
                        self.last_error = f"{new_key} is already loaded"
                        reward -= 10.0

                    else:
                        new_data = self.roster[new_key]
                        host_size = new_data["host_mb"]
                        effective_free = self.HARDWARE_RAM_MB - self.ram_used_mb - self.active_spike_mb - self.SYSTEM_OVERHEAD_MB

                        if host_size > effective_free:
                            self.last_error = f"OOM: {new_key} needs {host_size:.0f}MB free"
                            reward -= 30.0
                            self.oom_errors += 1
                        else:
                            reward += 5.0

                            is_warm = new_key in self.evicted_cache
                            base_load_s = new_data["load_avg_ms"] / 1000.0
                            if is_warm:
                                actual_load_s = base_load_s * random.uniform(0.1, 0.2)
                                self.evicted_cache.pop(new_key)
                            else:
                                mu = math.log(max(base_load_s, 1e-6))
                                actual_load_s = random.lognormvariate(mu, 0.15)

                            load_steps = max(1, math.ceil(actual_load_s))
                            for _ in range(load_steps - 1):
                                _clock_tick()

                            self.loaded_models[new_key] = {
                                "model": action.model_id,
                                "quant": action.quant_type,
                                "tier": new_data["tier"],
                                "size_mb": round(host_size),
                                "weight_mb": round(new_data["size_mb"]),
                                "cpu_avg": new_data["cpu_avg"],
                                "gen_tps": new_data["gen_tps"],
                                "prompt_tps": new_data["prompt_tps"],
                                "ctx_mb": new_data["ctx_mb"],
                                "comp_mb": new_data["comp_mb"],
                                "kv_mb_max": new_data["kv_mb_max"],
                            }
                            self.ram_used_mb += host_size
                            self.load_count += 1

                            reward -= actual_load_s * 1.5

                            after_loaded_count = len(self.loaded_models)
                            if before_loaded_count < 2 and after_loaded_count >= 2:
                                reward += 1.5

                            self.last_feedback = f"REPLACE: Evicted {evict_key}, Loaded {new_key} in {actual_load_s:.1f}s."
            else:
                self.last_error = f"Cannot replace {evict_key or 'nothing'}"
                reward -= 5.0

        if not self.queue:
            done = True
            reward += 50.0
            self.last_feedback = "SUCCESS: Queue cleared!"
        elif self.step_count >= self.MAX_STEPS:
            done = True
            reward -= 50.0

        if self.last_error:
            self.last_feedback = f"ERROR: {self.last_error}"

        self.cumulative_reward += reward
        obs = self._get_observation()
        obs.done = done
        obs.reward = reward
        return obs

    def _get_observation(self) -> ModelFlowObservation:
        model_summary = {}
        for role, model_id in self.role_to_model.items():
            stats = {}
            for quant in self.available_quants.get(model_id, []):
                key = f"{model_id}-{quant}"
                if key in self.roster:
                    d = self.roster[key]
                    stats[quant] = {
                        "size_mb": round(d["host_mb"]),
                        "gen_tps": d["gen_tps"],
                        "prompt_tps": d["prompt_tps"],
                        "cpu_avg": d["cpu_avg"],
                        "tier": d["tier"],
                    }
            model_summary[role] = {"model_id": model_id, "stats": stats}

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
            info={
                "cumulative_reward": round(self.cumulative_reward, 2),
                "completed": self.completed,
                "pending": len(self.queue),
                "ram_free_mb": round(self.HARDWARE_RAM_MB - self.ram_used_mb - self.active_spike_mb),
                "grader_metrics": {
                    "loads": self.load_count,
                    "evicts": self.evict_count,
                    "idles": self.idle_steps,
                    "ooms": self.oom_errors,
                    "reasoning_done": self.reasoning_completed
                }
            },
            reward=0.0
        )


    def state(self):
        return {
            "step": self.step_count,
            "ram_used_mb": self.ram_used_mb,
            "pending": len(self.queue),
            "loaded_models": list(self.loaded_models.keys()),
            "active_spike_mb": self.active_spike_mb,
            "spike_steps_remaining": self.spike_steps_remaining,
        }

    def get_episode_result(self) -> any:
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
            # ── NEW ──────────────────────────────────────────────────────────────
            completion_ages=self.completion_ages.copy(),
            throughput_samples=self.throughput_samples.copy(),
            overprovision_count=self.overprovision_count,
        )

    def score_task(self) -> float:
        from graders import grade
        result = self.get_episode_result()
        return grade(self.current_task, result)