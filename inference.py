"""
inference.py — ModelFlow

"""

import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

from helpers.queue_utils import parse_action, queue_stats
from models import ModelFlowAction
from prompt import build_messages, build_roster_str, get_system_prompt, observation_to_text
from server.modelflow_environment import ModelFlowEnvironment
from config import (
    BASE_BACKOFF_S,
    BENCHMARK,
    MAX_RETRIES,
    MAX_STEPS_PER_TASK,
    TASKS,
    TEMPERATURE,
    MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------

# USE_GROQ     = os.getenv("GROQ", "0") == "1"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# if USE_GROQ:
#     if not GROQ_API_KEY:
#         raise ValueError("GROQ_API_KEY is required when GROQ=1")
#     print("[INFO] Using Groq backend", flush=True)
#     client = OpenAI(
#         base_url="https://api.groq.com/openai/v1",
#         api_key=GROQ_API_KEY,
#         timeout=30.0,
#     )
#     if MODEL_NAME == "Qwen/Qwen2.5-72B-Instruct":
#         MODEL_NAME = "llama-3.3-70b-versatile"
# else:
#     if HF_TOKEN is None:
#         raise ValueError("HF_TOKEN environment variable is required")
#     print("[INFO] Using HuggingFace router backend", flush=True)
#     client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30.0)

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
print("[INFO] Using HuggingFace router backend", flush=True)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30.0)


EPISODE_LOG_PATH = Path(os.getenv("EPISODE_LOG_PATH", "episode_log.jsonl"))


# ---------------------------------------------------------------------------
# Output helpers — single source of truth for the three required line types
# ---------------------------------------------------------------------------

def _log_start(task_name: str) -> None:
    """
    [START] task=<task_name> env=<benchmark> model=<model_name>
    """
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(
    step_num:     int,
    final_action: ModelFlowAction,
    reward_val:   float,
    done:         bool,
    error:        Optional[str],
) -> None:
    """
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>

    action_str format: COMMAND(model_id,quant_type)
    e.g. EXECUTE(gemma-3-4b,Q4_K_M)  /  LOAD(llama_1b,Q6_K)  /  IDLE(-,-)
    """
    mid   = final_action.model_id   or "-"
    quant = final_action.quant_type or "-"
    action_str = f"{final_action.command}({mid},{quant})"

    print(
        f"[STEP] step={step_num}"
        f" action={action_str}"
        f" reward={reward_val:.2f}"
        f" done={'true' if done else 'false'}"
        f" error={error or 'null'}",
        flush=True,
    )


def _log_end(
    success:     bool,
    step_num:    int,
    final_score: float,
    rewards:     List[float],
) -> None:
    """
    [END] success=<true|false> steps=<n> score=<grader_score> rewards=<r1,r2,...>

    score= is the grader result for the completed task (0.00–1.00).
    rewards= is the comma-separated per-step reward list, each to 2 dp.
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'}"
        f" steps={step_num}"
        f" score={final_score:.2f}"
        f" rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# DecisionMemory
# ---------------------------------------------------------------------------

def _reward_band(reward: float) -> str:
    if reward > 10:
        return "good"
    if reward < -5:
        return "bad"
    return "neutral"


@dataclasses.dataclass
class _Entry:
    step:        int
    command:     str
    model_id:    Optional[str]
    quant_type:  Optional[str]
    band:        str
    result:      str
    swap_helped: Optional[bool]


class DecisionMemory:
    MAX = 5

    def __init__(self):
        self._entries:     List[_Entry] = []
        self._all_entries: List[_Entry] = []   # full history, never trimmed

    def push(
        self,
        step:     int,
        action:   ModelFlowAction,
        reward:   float,
        feedback: Optional[str],
        error:    Optional[str],
    ) -> None:
        band   = _reward_band(reward)
        result = "ok" if not error else error[:60]

        if self._entries:
            prev = self._entries[-1]
            if prev.swap_helped is None and prev.command in ("REPLACE", "EVICT"):
                updated = dataclasses.replace(prev, swap_helped=(band != "bad"))
                self._entries[-1]     = updated
                if self._all_entries:
                    self._all_entries[-1] = updated

        entry = _Entry(step, action.command, action.model_id, action.quant_type, band, result, None)
        self._entries.append(entry)
        self._all_entries.append(entry)

        if len(self._entries) > self.MAX:
            self._entries.pop(0)

    @property
    def bad_count_recent(self) -> int:
        return sum(1 for e in self._entries[-3:] if e.band == "bad")


# ---------------------------------------------------------------------------
# Policy filter — blocks logically impossible or provably self-defeating actions
# ---------------------------------------------------------------------------

def _find_loaded_quant_for_model(model_id: str, obs) -> Optional[str]:
    """Return the quant currently loaded for model_id, or None."""
    for key, slot in obs.loaded_models.items():
        if slot.get("model") == model_id:
            return slot.get("quant")
    return None


def _policy_filter(
    raw_action: ModelFlowAction,
    obs,
) -> Tuple[ModelFlowAction, Optional[str]]:
    """
    Returns (final_action, override_reason_or_None).

    Intercepts:
    1. Missing model_id on commands that require it → IDLE.
    2. LOAD of an already-loaded key → redirect to EXECUTE.
    3. EXECUTE with a quant not matching the loaded slot for that model
       → redirect to EXECUTE with the actually-loaded quant, or IDLE if
       the model isn't loaded at all.
    4. REPLACE where the *new* model+quant is already loaded → redirect
       to EXECUTE (the evict half is skipped entirely, saving RAM churn).
    5. REPLACE where evict_key == new_key (self-replace) → redirect to
       EXECUTE if that key is loaded, else IDLE.

    Everything else passes through to the environment for real feedback.
    """
    # ── 1. Missing model_id ─────────────────────────────────────────────────
    if raw_action.command in ("LOAD", "EXECUTE", "REPLACE") and not raw_action.model_id:
        return (
            ModelFlowAction(
                command="IDLE",
                model_id=None, quant_type=None, batch_size=1,
                evict_model_id=None, evict_quant_type=None,
            ),
            f"{raw_action.command} missing model_id → IDLE",
        )

    # ── 2. LOAD of already-loaded key → EXECUTE ─────────────────────────────
    if raw_action.command == "LOAD" and raw_action.model_id and raw_action.quant_type:
        key = f"{raw_action.model_id}-{raw_action.quant_type}"
        if key in obs.loaded_models:
            return (
                ModelFlowAction(
                    command="EXECUTE",
                    model_id=raw_action.model_id,
                    quant_type=raw_action.quant_type,
                    batch_size=min(8, max(1, raw_action.batch_size or 1)),
                    evict_model_id=None,
                    evict_quant_type=None,
                ),
                f"LOAD blocked: {key} already loaded → redirected to EXECUTE",
            )

    # ── 3. EXECUTE with wrong/unloaded quant for that model ─────────────────
    if raw_action.command == "EXECUTE" and raw_action.model_id and raw_action.quant_type:
        requested_key = f"{raw_action.model_id}-{raw_action.quant_type}"
        if requested_key not in obs.loaded_models:
            # Check if the model is loaded under a different quant.
            actual_quant = _find_loaded_quant_for_model(raw_action.model_id, obs)
            if actual_quant is not None:
                corrected_key = f"{raw_action.model_id}-{actual_quant}"
                return (
                    ModelFlowAction(
                        command="EXECUTE",
                        model_id=raw_action.model_id,
                        quant_type=actual_quant,
                        batch_size=min(8, max(1, raw_action.batch_size or 1)),
                        evict_model_id=None,
                        evict_quant_type=None,
                    ),
                    (
                        f"EXECUTE blocked: {requested_key} not loaded"
                        f" but {corrected_key} is → redirected to EXECUTE with loaded quant"
                    ),
                )
            # Model not loaded at all — let env give the real error (costs -10).
            # Don't redirect to IDLE; the real penalty teaches the agent.
            return raw_action, None

    # ── 4. REPLACE where the *new* model+quant is already loaded ────────────
    if raw_action.command == "REPLACE" and raw_action.model_id and raw_action.quant_type:
        new_key = f"{raw_action.model_id}-{raw_action.quant_type}"
        if new_key in obs.loaded_models:
            # The target is already loaded — just execute it.
            return (
                ModelFlowAction(
                    command="EXECUTE",
                    model_id=raw_action.model_id,
                    quant_type=raw_action.quant_type,
                    batch_size=min(8, max(1, raw_action.batch_size or 1)),
                    evict_model_id=None,
                    evict_quant_type=None,
                ),
                (
                    f"REPLACE blocked: {new_key} is already loaded"
                    f" → redirected to EXECUTE (skipping destructive evict)"
                ),
            )

        # ── 5. Self-replace: evict_key == new_key ───────────────────────────
        if raw_action.evict_model_id and raw_action.evict_quant_type:
            evict_key = f"{raw_action.evict_model_id}-{raw_action.evict_quant_type}"
            if evict_key == new_key:
                # Agent wants to evict and reload the exact same model+quant.
                if new_key in obs.loaded_models:
                    return (
                        ModelFlowAction(
                            command="EXECUTE",
                            model_id=raw_action.model_id,
                            quant_type=raw_action.quant_type,
                            batch_size=min(8, max(1, raw_action.batch_size or 1)),
                            evict_model_id=None,
                            evict_quant_type=None,
                        ),
                        f"Self-replace blocked: evict+load of same key {new_key} → EXECUTE",
                    )
                else:
                    return (
                        ModelFlowAction(
                            command="IDLE",
                            model_id=None, quant_type=None, batch_size=1,
                            evict_model_id=None, evict_quant_type=None,
                        ),
                        f"Self-replace blocked: evict+load of same key {new_key} (not loaded) → IDLE",
                    )

    return raw_action, None


def _print_decision_debug(obs, raw_action, final_action, override_reason):
    print("\n[DECISION TRACE]")
    print(f"  Proposed : {raw_action.command} {raw_action.model_id or ''}/{raw_action.quant_type or ''}")
    print(f"  Final    : {final_action.command} {final_action.model_id or ''}/{final_action.quant_type or ''}")
    if override_reason:
        print(f"  OVERRIDE : {override_reason}")
    print(f"  Loaded   : {list(obs.loaded_models.keys())}")
    print(f"  Last err : {obs.last_action_error or 'None'}", flush=True)


# ---------------------------------------------------------------------------
# Episode logger
# ---------------------------------------------------------------------------

def _extract_mistakes(memory: DecisionMemory, rewards: List[float]) -> List[str]:
    """
    Build distinct error patterns from the FULL episode history.
    Pairs _all_entries with rewards by step number alignment.
    """
    mistakes = []
    seen: set = set()

    step_to_reward: dict = {}
    for i, e in enumerate(memory._all_entries):
        if i < len(rewards):
            step_to_reward[e.step] = rewards[i]

    for e in memory._all_entries:
        r = step_to_reward.get(e.step, 0.0)
        if _reward_band(r) == "bad":
            pattern = f"{e.command} {e.model_id or '-'}/{e.quant_type or '-'}: {e.result}"
            if pattern not in seen:
                seen.add(pattern)
                mistakes.append(pattern)

    return mistakes[:5]


def _write_episode_log(
    task_name: str,
    score:     float,
    steps:     int,
    rewards:   List[float],
    memory:    DecisionMemory,
    success:   bool,
) -> None:
    top_errors  = _extract_mistakes(memory, rewards)
    bad_entries = [
        e for i, e in enumerate(memory._all_entries)
        if i < len(rewards) and _reward_band(rewards[i]) == "bad"
    ]

    # Detect quant overprovisioning pattern.
    q6k_on_std = sum(
        1 for e in memory._all_entries
        if e.command == "EXECUTE" and e.quant_type in ("Q6_K", "Q8_0")
    )
    exec_total = sum(1 for e in memory._all_entries if e.command == "EXECUTE")
    quant_note = (
        f"Q6_K/Q8_0 used on {q6k_on_std}/{exec_total} EXECUTE steps"
        f" — check if standard-only requests were overprovisioned."
        if exec_total > 0 and q6k_on_std > exec_total // 2
        else ""
    )

    if not bad_entries:
        summary = "No major mistakes."
    else:
        cmd_counts: dict = {}
        for e in bad_entries:
            cmd_counts[e.command] = cmd_counts.get(e.command, 0) + 1
        worst   = max(cmd_counts, key=cmd_counts.get)
        bad_avg = (
            sum(rewards[i] for i, e in enumerate(memory._all_entries)
                if i < len(rewards) and _reward_band(rewards[i]) == "bad")
            / max(len(bad_entries), 1)
        )
        summary = f"Most frequent bad action: {worst} ({cmd_counts[worst]}x), avg_reward={bad_avg:.1f}."
        if quant_note:
            summary += " " + quant_note

    record = {
        "task":            task_name,
        "score":           round(score, 4),
        "steps":           steps,
        "success":         success,
        "mean_reward":     round(sum(rewards) / max(len(rewards), 1), 2),
        "bad_step_count":  len(bad_entries),
        "top_errors":      top_errors,
        "mistake_summary": summary,
    }

    try:
        with EPISODE_LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[WARN] Episode log write failed: {exc}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> None:
    env    = ModelFlowEnvironment()
    memory = DecisionMemory()

    obs:         object       = None
    step_num:    int          = 0
    rewards:     List[float]  = []
    done:        bool         = False
    final_score: float        = 0.0

    prev_overridden:    bool          = False
    prev_override_from: Optional[str] = None

    # ── [START] ──────────────────────────────────────────────────────────────
    _log_start(task_name)

    try:
        obs = env.reset(task_name=task_name)

        while not done and step_num < MAX_STEPS_PER_TASK:
            step_num += 1

            q_stats_now   = queue_stats(obs)
            roster_str    = build_roster_str(obs)
            system_prompt = get_system_prompt(roster_str, obs.ram_limit_mb, q_stats_now, obs)

            obs_text = observation_to_text(
                obs,
                q_stats_now,
                memory,
                last_was_overridden=prev_overridden,
                overridden_from=prev_override_from,
            )

            messages = build_messages(system_prompt, obs_text)

            # LLM call with exponential backoff on rate limits.
            action_dict = {
                "command": "IDLE", "model_id": None, "quant_type": None,
                "batch_size": 8, "evict_model_id": None, "evict_quant_type": None,
            }
            llm_success = False
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
                    llm_success = True
                    break
                except Exception as exc:
                    err_str = str(exc)
                    print(f"[LLM ERROR] attempt={attempt+1} {err_str}", file=sys.stderr, flush=True)
                    if ("429" in err_str or "rate limit" in err_str.lower()) and attempt < MAX_RETRIES - 1:
                        time.sleep(BASE_BACKOFF_S * (2 ** attempt))
                        continue
                    break

            if not llm_success:
                print("[WARN] All LLM retries failed — using IDLE", file=sys.stderr, flush=True)

            raw_action = ModelFlowAction(
                command=action_dict.get("command", "IDLE"),
                model_id=action_dict.get("model_id"),
                quant_type=action_dict.get("quant_type"),
                batch_size=min(action_dict.get("batch_size", 8), 8),
                evict_model_id=action_dict.get("evict_model_id"),
                evict_quant_type=action_dict.get("evict_quant_type"),
            )

            final_action, override_reason = _policy_filter(raw_action, obs)

            _print_decision_debug(obs, raw_action, final_action, override_reason)

            obs  = env.step(final_action)
            done = obs.done

            reward_val = round(obs.reward, 2)
            rewards.append(reward_val)

            memory.push(
                step_num, final_action, reward_val,
                obs.last_action_feedback, obs.last_action_error,
            )

            if override_reason:
                prev_overridden    = True
                prev_override_from = (
                    f"{raw_action.command} {raw_action.model_id or ''}/{raw_action.quant_type or ''}"
                )
            else:
                prev_overridden    = False
                prev_override_from = None

            # ── [STEP] ───────────────────────────────────────────────────────
            _log_step(step_num, final_action, reward_val, done, obs.last_action_error)

        final_score = env.score_task()

    except Exception as exc:
        print(f"[EXCEPTION] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

    finally:
        success = (
            obs is not None
            and not obs.queue
            and getattr(obs, "deferred_count", 0) == 0
        )

        # ── [END] ────────────────────────────────────────────────────────────
        _log_end(success, step_num, final_score, rewards)

        _write_episode_log(
            task_name=task_name,
            score=final_score,
            steps=step_num,
            rewards=rewards,
            memory=memory,
            success=success,
        )


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
