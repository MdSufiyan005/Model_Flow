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
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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
=== DECISION RULES — FOLLOW IN ORDER EVERY STEP ===

STEP 1 — READ THE QUEUE.
  Identify: which model_id is needed, total count, and whether ANY requests are "reasoning".

STEP 2 — PICK THE CORRECT QUANT.
  - All requests are "standard" or "simple"  →  use Q4_K_M (smallest, cheapest)
  - ANY request is "reasoning"               →  use Q6_K minimum
  DO NOT upgrade quant unless reasoning requests are actually present.
  DO NOT replace a model you just loaded this step.

STEP 3 — CHECK RAM BEFORE EVERY LOAD OR EXECUTE.
  effective_free = ram_limit_mb - ram_used_mb - pressure_spike_mb - {RAM_SAFETY_BUFFER_MB}
  Only LOAD if: model_size_mb < effective_free
  If spike is active: use batch_size = min(2, matching_queue_entries)
  Otherwise:          use batch_size = min(8, matching_queue_entries)

STEP 4 — EXECUTE THE LOADED MODEL UNTIL ITS QUEUE IS EMPTY.
  If the correct model is already loaded at a sufficient quant → go straight to EXECUTE.
  Only REPLACE if a completely different role's model needs to be served next.

STEP 5 — FINISH IN ≤8 STEPS.
  single-load task = one model type only. Load once, execute until done. Never replace.

OUTPUT — ONE JSON object only, no markdown, no explanation:
{{"command":"LOAD|EXECUTE|EVICT|REPLACE|IDLE","model_id":"...","quant_type":"...","batch_size":8,"evict_model_id":null,"evict_quant_type":null}}
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
            # ── Hard batch_size cap based on free RAM ──────────────────────
            # ── Hard batch_size cap based on actual free RAM ───────────────
            if action.command == "EXECUTE" and action.model_id and action.quant_type:
                key = f"{action.model_id}_{action.quant_type}"
                slot = obs.loaded_models.get(key, {})
                model_mb = slot.get("size_mb", 0)
                # Use only attributes that exist on ModelFlowObservation
                actual_free = obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb
                # Rough KV-cache estimate per batch item
                kv_per_item = max(80, model_mb // 10)
                safe_batch = max(1, int((actual_free - RAM_SAFETY_BUFFER_MB) // kv_per_item))
                safe_batch = min(safe_batch, action.batch_size)
                if safe_batch < action.batch_size:
                    print(
                        f"[OOM GUARD] batch {action.batch_size}→{safe_batch} "
                        f"(actual_free={actual_free}MB kv≈{kv_per_item}MB/item)",
                        file=sys.stderr, flush=True,
                    )
                    action = ModelFlowAction(
                        command=action.command,
                        model_id=action.model_id,
                        quant_type=action.quant_type,
                        batch_size=safe_batch,
                        evict_model_id=action.evict_model_id,
                        evict_quant_type=action.evict_quant_type,
                    )            
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
    for task in TASKS:
        run_task(task)
    # run_task("quality-limit")