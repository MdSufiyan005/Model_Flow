from sys import stderr
import sys
import time
from typing import List

from config import (
    BENCHMARK,
    BASE_BACKOFF_S,
    CONTEXT_HISTORY_STEPS,
    MAX_RETRIES,
    MAX_STEPS_PER_TASK,
    TASKS,
    active_model,
)
from helpers.context_utils import compress_step
from helpers.llm_utils import llm_call
from helpers.planning import apply_planning_override
from helpers.queue_utils import parse_action, queue_stats
from helpers.visualization import print_visualization
from models import ModelFlowAction
from prompt import build_messages, build_roster_str, get_system_prompt, observation_to_text
from server.modelflow_environment import ModelFlowEnvironment

def run_task(task_name: str):
    env = ModelFlowEnvironment()
    obs = env.reset(task_name=task_name)

    step_num = 0
    rewards = []
    done = False
    compressed_history: List[str] = []
    score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={active_model}", flush=True)
    print_visualization(task_name, 0, obs)

    try:
        while not done and step_num < MAX_STEPS_PER_TASK:
            step_num += 1

            q_stats_now = queue_stats(obs)
            roster_str = build_roster_str(obs)
            system_prompt = get_system_prompt(roster_str, obs.ram_limit_mb, q_stats_now, obs)
            obs_text = observation_to_text(obs, q_stats_now)
            recent = compressed_history[-CONTEXT_HISTORY_STEPS:]
            messages = build_messages(system_prompt, recent, obs_text)

            action_dict = {
                "command": "IDLE",
                "model_id": None,
                "quant_type": None,
                "batch_size": 1,
                "evict_model_id": None,
                "evict_quant_type": None,
            }

            for attempt in range(MAX_RETRIES):
                try:
                    raw = llm_call(messages)
                    print(f"[LLM RAW]: {raw}", file=sys.stderr)
                    action_dict = parse_action(raw)
                    break
                except Exception as e:
                    err_str = str(e)
                    wait = BASE_BACKOFF_S * (2 ** attempt)
                    if any(kw in err_str.lower() for kw in ("rate limit", "429", "too many")) and attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                    else:
                        break

            action = ModelFlowAction(
                command=action_dict.get("command", "IDLE"),
                model_id=action_dict.get("model_id"),
                quant_type=action_dict.get("quant_type"),
                batch_size=min(action_dict.get("batch_size", 8), 8),
                evict_model_id=action_dict.get("evict_model_id"),
                evict_quant_type=action_dict.get("evict_quant_type"),
            )
            action = apply_planning_override(action, obs)

            obs = env.step(action)
            reward_val = round(obs.reward, 2)
            rewards.append(reward_val)
            done = obs.done

            print_visualization(task_name, step_num, obs, action, reward_val)

            done_str = "true" if done else "false"
            error_val = obs.last_action_error or "null"
            requested_batches = action.batch_size or 1
            executed_batches = requested_batches if action.command == "EXECUTE" and not obs.last_action_error else 0
            not_executed_batches = requested_batches - executed_batches

            print(
                f"[STEP] step={step_num} "
                f"action={action.command}[{action.model_id or ''}-{action.quant_type}, "
                f"batch={requested_batches}, executed={executed_batches}, not_executed={not_executed_batches}] "
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

        score = env.score_task()

    finally:
        success_str = "true" if len(obs.queue) == 0 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}",
            flush=True,
        )

if __name__ == "__main__":
    import sys
    import traceback

    for task in TASKS:
        try:
            run_task(task)
        except Exception as e:
            print(f"[ERROR] task={task} failed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()