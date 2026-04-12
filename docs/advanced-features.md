# Cross-Episode Learning, Decision Tree & Tick Caps

## Cross-Episode Learning
After each episode, inference.py writes a JSON record to episode_log.jsonl containing the task name, score, step count, bad-step count, top error patterns, and a mistake summary (most frequent bad action, quant over-provisioning note if applicable).

At the start of each episode, the prompt layer reads the last 3 records for the same task and injects them as a LESSONS FROM PAST EPISODES block at the top of the system prompt. Task filtering is intentional: quality-limit lessons (which warn against Q6_K overuse) would actively harm ram-pressure performance where Q6_K is required for reasoning.

## Decision Tree (from System Prompt)
The agent is instructed to follow this logic every step:

1. Read REMAINING — identify which models still have pending requests. If all counts are zero, the episode should be ending.
2. Check HINTS — find any model where rec_already_loaded=True AND servable > 0. If found → EXECUTE (Step 3).
3. EXECUTE — use the exact loaded quant, batch_size = min(8, servable) (or min 2 during a RAM spike).
4. Load the right model — pick the BEST TARGET from HINTS (must appear in REMAINING). Check exec_safe: if True use recommended_quant, else use safe_quant. Then: if it fits RAM and isn't loaded → LOAD; if RAM is tight → REPLACE, evicting the model with count=0 in REMAINING.
5. IDLE — last resort only, costs −15.

## Tick Caps
To keep the reward signal learnable, internal simulation loops are bounded:

* LOAD/REPLACE load phase: max 4 clock ticks regardless of actual load time.
* EXECUTE: max 8 clock ticks regardless of actual inference time.

Actual wall-clock time (total_time_s) is still computed accurately and reported in feedback for the agent to reason about throughput; only the age-penalty accumulation is capped.
