"""
inference.py — ModelFlow benchmark agent.
"""

import os
import sys
import time
import traceback
from typing import List

from openai import OpenAI

# ────────────────────────────────────────────────────────────────
# Environment / Config
# ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:together")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
    timeout=30.0
)

# ────────────────────────────────────────────────────────────────
# Internal Imports
# ────────────────────────────────────────────────────────────────
from config import (
    BENCHMARK,
    BASE_BACKOFF_S,
    CONTEXT_HISTORY_STEPS,
    MAX_RETRIES,
    MAX_STEPS_PER_TASK,
    RAM_SAFETY_BUFFER_MB,
    TASKS,
    TEMPERATURE,
    MAX_TOKENS,
)

from helpers.context_utils import compress_step
from helpers.planning import apply_planning_override
from helpers.queue_utils import parse_action, queue_stats
from models import ModelFlowAction
from prompt import (
    build_messages,
    build_roster_str,
    get_system_prompt,
    observation_to_text,
)
from server.modelflow_environment import ModelFlowEnvironment

# ────────────────────────────────────────────────────────────────
# Rules
# ────────────────────────────────────────────────────────────────
_TIER_RULES = f"""
=== HARD CONSTRAINTS — NEVER VIOLATE THESE ===

QUANT → TIER → RANK:
Q4_K_M → low → rank 1 (simple requests ONLY)
Q5_K_M → medium → rank 2 (simple requests ONLY)
Q6_K → high → rank 3 (simple + reasoning ✓)
Q8_0 → risky → rank 4 (simple + reasoning ✓)

RULE 1 Any "reasoning" request requires Q6_K or Q8_0.
Never LOAD Q4_K_M or Q5_K_M if that model has ANY reasoning requests queued.
If the wrong (too-low) quant is already loaded, use REPLACE to upgrade.

RULE 2 Prefer REPLACE over separate EVICT then LOAD — it saves a step.

RULE 3 RAM guard: before LOAD, check ram_free_mb.
Must satisfy: model_size_mb + {RAM_SAFETY_BUFFER_MB} <= ram_free_mb.
If a pressure spike is active, also subtract pressure_spike_mb.

RULE 4 Never IDLE when queue is non-empty and a model is loaded.

RULE 5 Always set batch_size = min(8, queue_length).

RULE 6 Finish in ≤8 steps if possible.

OUTPUT — respond with ONLY a single JSON object:
{{
    "command": "LOAD | EXECUTE | EVICT | REPLACE | IDLE",
    "model_id": "<model_id or null>",
    "quant_type": "<Q4_K_M | Q5_K_M | Q6_K | Q8_0 or null>",
    "batch_size": <1-8>,
    "evict_model_id": "<model_id or null>",
    "evict_quant_type": "<quant or null>"
}}
"""

# ────────────────────────────────────────────────────────────────
# Prompt Builder
# ────────────────────────────────────────────────────────────────
def _build_system_prompt(roster_str: str, ram_limit: int, q_stats: dict, obs) -> str:
    base = get_system_prompt(roster_str, ram_limit, q_stats, obs)
    return base + "\n" + _TIER_RULES


# ────────────────────────────────────────────────────────────────
# Core Runner
# ────────────────────────────────────────────────────────────────
def run_task(task_name: str) -> None:
    env = ModelFlowEnvironment()

    obs = None
    step_num = 0
    rewards: List[float] = []
    done = False
    compressed_history: List[str] = []
    final_score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_name=task_name)

        while not done and step_num < MAX_STEPS_PER_TASK:
            step_num += 1

            # ── Prompt Construction ────────────────────────────────
            q_stats_now = queue_stats(obs)
            roster_str = build_roster_str(obs)

            system_prompt = _build_system_prompt(
                roster_str,
                obs.ram_limit_mb,
                q_stats_now,
                obs,
            )

            obs_text = observation_to_text(obs, q_stats_now)
            recent = compressed_history[-CONTEXT_HISTORY_STEPS:]

            messages = build_messages(system_prompt, recent, obs_text)

            # ── LLM Call ───────────────────────────────────────────
            action_dict = {
                "command": "IDLE",
                "model_id": None,
                "quant_type": None,
                "batch_size": 8,
                "evict_model_id": None,
                "evict_quant_type": None,
            }

            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )

                    raw = (response.choices[0].message.content or "").strip()
                    action_dict = parse_action(raw)
                    break

                except Exception as exc:
                    err_str = str(exc)
                    print(
                        f"[LLM ERROR] attempt={attempt + 1} error={err_str}",
                        file=sys.stderr,
                        flush=True,
                    )

                    if any(k in err_str.lower() for k in ("rate limit", "429", "too many")):
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(BASE_BACKOFF_S * (2 ** attempt))
                            continue
                    break

            # ── Build Action ───────────────────────────────────────
            raw_action = ModelFlowAction(
                command=action_dict.get("command", "IDLE"),
                model_id=action_dict.get("model_id"),
                quant_type=action_dict.get("quant_type"),
                batch_size=min(action_dict.get("batch_size", 8), 8),
                evict_model_id=action_dict.get("evict_model_id"),
                evict_quant_type=action_dict.get("evict_quant_type"),
            )

            # ── Planning Override ──────────────────────────────────
            action = apply_planning_override(raw_action, obs)

            if (
                action.command != raw_action.command
                or action.quant_type != raw_action.quant_type
            ):
                print(
                    f"[PLANNING] overrode {raw_action.command}/{raw_action.quant_type} "
                    f"→ {action.command}/{action.quant_type}",
                    file=sys.stderr,
                    flush=True,
                )

            # ── Runtime OOM Guard ──────────────────────────────────
            if action.command == "EXECUTE" and action.model_id:
                from helpers.queue_utils import loaded_key as lk_fn

                lk = lk_fn(obs, action.model_id)
                if lk:
                    _ = obs.loaded_models.get(lk, {})

            # ── Step Environment ───────────────────────────────────
            obs = env.step(action)
            done = obs.done

            reward_val = round(obs.reward, 2)
            rewards.append(reward_val)

            error_val = obs.last_action_error or "null"
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={step_num} action={action.command} "
                f"reward={reward_val:.2f} done={done_str} error={error_val}",
                flush=True,
            )

            compressed_history.append(
                compress_step(
                    step_num,
                    action,
                    obs.reward,
                    obs.last_action_feedback,
                    obs.last_action_error,
                )
            )

        final_score = env.score_task()

    except Exception as exc:
        print(f"[EXCEPTION] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

    finally:
        success_str = "true" if (obs is not None and len(obs.queue) == 0) else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

        print(
            f"[END] success={success_str} steps={step_num} "
            f"score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )


# ────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # for task in TASKS:
    #     run_task(task)
    run_task("quality-limit")