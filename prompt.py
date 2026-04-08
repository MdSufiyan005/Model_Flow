from typing import Dict, List

from config import REASONING_MIN_QUANT, SYSTEM_OVERHEAD_MB
from helpers.model_utils import ram_free
from helpers.queue_utils import loaded_key, queue_stats
from helpers.tools import can_load as tool_can_load, simulate_execute_peak
from models import ModelFlowObservation


def build_roster_str(obs: ModelFlowObservation) -> str:
    lines = ["  MODEL-QUANT          | TIER   | SIZE(MB) | GEN t/s | PROMPT t/s"]
    for role, info in obs.model_summary.items():
        for quant, stats in sorted(info["stats"].items()):
            lines.append(
                f"  {info['model_id']}-{quant:<10} | {stats['tier']:<6} | "
                f"{stats['size_mb']:>6}   | {stats['gen_tps']:>7.1f} | {stats['prompt_tps']:>9.1f}"
            )
    return "\n".join(lines)


def get_system_prompt(
    roster_str: str,
    ram_limit_mb: int,
    q_stats: Dict[str, Dict],
    obs: ModelFlowObservation,
) -> str:
    """
    Concise system prompt. RAM arithmetic is offloaded to tools so the prompt
    no longer needs to spell out formulas — just rules.
    """
    free_mb = ram_free(obs)
    loaded_parts = [f"{k}: {v['size_mb']}MB tier={v['tier']}" for k, v in obs.loaded_models.items()]
    loaded_str = ", ".join(loaded_parts) if loaded_parts else "none"

    work_lines = [
        f"  {mid}: {qs['total']} requests ({qs['reasoning']} reasoning, {qs['standard']} standard)"
        for mid, qs in q_stats.items()
    ]
    work_str = "\n".join(work_lines) if work_lines else "  (queue empty)"

    return f"""You are an ML infrastructure orchestrator. Clear ALL queued requests efficiently.

HARDWARE
  RAM limit: {ram_limit_mb} MB  |  Effective free now: {free_mb} MB
  Currently loaded: {loaded_str}

ROLE → MODEL
  chatbot → gemma-3-4b  |  translator → llama_1b  |  coder → qwen3.5-2b

QUANT TIERS
  Q4_K_M / Q5_K_M = standard only  |  Q6_K / Q8_0 = standard + reasoning

DECISION RULES
  1. Read RAM ANALYSIS in the observation — it already tells you if a LOAD fits and if EXECUTE is safe.
  2. Any model with reasoning requests → Q6_K minimum.
  3. batch_size = all pending requests for that model (max 8).
  4. EXECUTE immediately after every LOAD / REPLACE.
  5. Use REPLACE when swapping quant on the same model (not EVICT+LOAD).
  6. Only evict after a model's queue section is fully drained.

PENDING WORK
{work_str}

AVAILABLE CONFIGS
{roster_str}

Respond ONLY with valid JSON (no markdown):
{{"command": "LOAD"|"EXECUTE"|"EVICT"|"REPLACE"|"IDLE",
  "model_id": str|null, "quant_type": str|null, "batch_size": int,
  "evict_model_id": str|null, "evict_quant_type": str|null}}"""


def observation_to_text(obs: ModelFlowObservation, q_stats: Dict[str, Dict]) -> str:
    if obs.last_action_error:
        outcome = f"LAST ACTION FAILED: {obs.last_action_error}"
    elif obs.last_action_feedback:
        outcome = f"LAST ACTION OK: {obs.last_action_feedback}"
    else:
        outcome = "READY."

    free_mb = ram_free(obs)
    spike_str = (
        f" | Spike +{obs.pressure_spike_mb}MB ({obs.spike_steps_remaining} steps)"
        if obs.pressure_spike_mb > 0
        else ""
    )
    ram_line = (
        f"RAM: {obs.ram_used_mb}/{obs.ram_limit_mb}MB used "
        f"(free={free_mb}MB){spike_str}"
    )

    sum_cpu = sum(m.get("cpu_avg", 0) for m in obs.loaded_models.values())
    contention = min(0.8, sum_cpu / 400.0)
    loaded = [
        f"{key}(tier={v['tier']}, {v['size_mb']}MB, {v.get('gen_tps',0)*(1-contention):.1f}t/s)"
        for key, v in obs.loaded_models.items()
    ]
    loaded_str = "LOADED: " + (", ".join(loaded) if loaded else "NONE")

    work_parts = [
        f"{mid}: {qs['total']} pending ({qs['reasoning']} reasoning)"
        for mid, qs in q_stats.items()
    ]
    work_str = "QUEUE: " + (" | ".join(work_parts) if work_parts else "EMPTY")

    top_items = ", ".join(
        f"{r.request_id}:{r.model_type[:4]}/{r.complexity[:3]}/age={r.age_steps}"
        for r in obs.queue[:8]
    )
    top_str = f"TOP8: [{top_items}]" if top_items else ""

    metrics = obs.info.get("grader_metrics", {})
    meta = (
        f"Step={obs.step_count} completed={obs.info.get('completed',0)} "
        f"loads={metrics.get('loads',0)} evicts={metrics.get('evicts',0)} "
        f"ooms={metrics.get('ooms',0)} idles={metrics.get('idles',0)}"
    )

    analysis_lines = ["RAM ANALYSIS:"]
    for mid, qs in q_stats.items():
        needed_q = REASONING_MIN_QUANT if qs.get("reasoning", 0) > 0 else "Q4_K_M"
        cl = tool_can_load(mid, needed_q, obs)
        if "error" not in cl:
            fits_str = "fits" if cl["fits"] else f"NO_FIT(need {cl['deficit_mb']}MB more)"
            analysis_lines.append(
                f"  LOAD {mid}-{needed_q}: {cl['host_mb']}MB → {fits_str}"
            )

        lkey = loaded_key(obs, mid)
        if lkey:
            lq = obs.loaded_models[lkey]["quant"]
            bs = min(qs.get("total", 1), 8)
            peak = simulate_execute_peak(mid, lq, bs, obs)
            if "error" not in peak:
                safe_str = "SAFE" if peak["safe"] else f"OOM_by_{-peak['headroom_mb']}MB"
                analysis_lines.append(
                    f"  EXEC {lkey} b={bs}: peak={peak['total_ram_peak_mb']}MB → {safe_str}"
                )

    analysis_str = "\n".join(analysis_lines) if len(analysis_lines) > 1 else ""

    return "\n".join(
        filter(None, [outcome, ram_line, loaded_str, work_str, top_str, meta, analysis_str])
    )


def build_messages(system_prompt, compressed_history, current_obs_text) -> List[Dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if compressed_history:
        history_block = "STEP HISTORY (recent):\n" + "\n".join(
            f"  {line}" for line in compressed_history
        )
        messages.append({"role": "user", "content": history_block})
        messages.append({"role": "assistant", "content": "Acknowledged."})
    messages.append({"role": "user", "content": current_obs_text})
    return messages