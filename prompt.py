"""
prompt.py

Builds the system prompt, observation text, and message list sent to the LLM.

Design goals
------------
* System prompt  — sent once, contains everything that never changes:
                   roster table, action docs, scoring rules.
* Obs text       — sent every step, contains ONLY what changes:
                   RAM state, loaded models, queue breakdown, last feedback.
* History block  — compressed prior steps, oldest→newest.

This keeps per-call token count low while ensuring the LLM has the right info.
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from models import ModelFlowObservation

from config import SYSTEM_OVERHEAD_MB, REASONING_MIN_QUANT, REASONING_QUANTS,RAM_SAFETY_BUFFER_MB


# ── Roster string — built once, embedded in system prompt ───────────────────

def build_roster_str(obs: "ModelFlowObservation") -> str:
    """
    Compact table: role | model_id | quant | size_mb | tier | can_reason
    The role→model_id mapping is the critical column the LLM must internalize.
    """
    lines = [
        "ROLE→MODEL MAPPING (memorise this — queue uses role names):",
        f"  {'role':12s} {'model_id':14s} {'quant':8s} {'size_mb':>8s} {'tier':8s} {'reasoning':>9s}",
        "  " + "-" * 68,
    ]
    for role, info in sorted(obs.model_summary.items()):
        model_id = info["model_id"]
        for quant, stats in sorted(info["stats"].items()):
            can = "YES" if quant in REASONING_QUANTS else "no"
            lines.append(
                f"  {role:12s} {model_id:14s} {quant:8s} {stats['size_mb']:>8d}MB"
                f"  {stats['tier']:8s} {can:>9s}"
            )
    lines.append("")
    lines.append("KEY RULE: a queue entry with model_type='chatbot' needs model_id='gemma-3-4b',")
    lines.append("          'translator' needs 'llama_1b', 'coder' needs 'qwen3.5-2b'.")
    lines.append("          NEVER execute a model that has zero matching queue entries.")
    return "\n".join(lines)


# ── Queue stats ──────────────────────────────────────────────────────────────

def _queue_summary(obs: "ModelFlowObservation", q_stats: Dict) -> str:
    if not obs.queue:
        return "QUEUE: empty"

    # Role→model lookup from obs
    role_to_model = {role: info["model_id"] for role, info in obs.model_summary.items()}

    lines = [f"QUEUE: {len(obs.queue)} pending"]

    # Per-model breakdown with required quant explicitly stated
    for model_id, stats in q_stats.items():
        r   = stats.get("reasoning", 0)
        s   = stats.get("standard",  0)
        req = REASONING_MIN_QUANT if r > 0 else "Q4_K_M"
        lines.append(f"  {model_id}: {s} standard + {r} reasoning  → need {req}+")

    # Show first 6 queue entries so LLM sees exact role sequence
    lines.append("  Next requests:")
    for req in obs.queue[:6]:
        mid = role_to_model.get(req.model_type, "?")
        lines.append(f"    [{req.model_type}→{mid}] {req.complexity}  age={req.age_steps}")
    if len(obs.queue) > 6:
        lines.append(f"    ... and {len(obs.queue)-6} more")

    return "\n".join(lines)


# ── Loaded models ────────────────────────────────────────────────────────────

def _loaded_summary(obs: "ModelFlowObservation") -> str:
    if not obs.loaded_models:
        return "LOADED: none"
    lines = ["LOADED:"]
    for key, slot in obs.loaded_models.items():
        can = "reasoning OK" if slot.get("quant") in REASONING_QUANTS else "simple only"
        lines.append(
            f"  {key}  tier={slot['tier']}  {slot['size_mb']}MB  {can}"
        )
    return "\n".join(lines)


# ── RAM line ─────────────────────────────────────────────────────────────────

def _ram_line(obs: "ModelFlowObservation") -> str:
    spike = obs.pressure_spike_mb
    effective_free = max(0, int(
        obs.ram_limit_mb - obs.ram_used_mb - spike - SYSTEM_OVERHEAD_MB
    ))
    safe_threshold = RAM_SAFETY_BUFFER_MB  # import this from config
    
    base = (
        f"RAM: used={obs.ram_used_mb}MB / limit={obs.ram_limit_mb}MB | "
        f"effective_free={effective_free}MB | "
        f"safe_to_load_if_model_size < {effective_free - safe_threshold}MB"
    )
    if spike > 0:
        base += (
            f"\n⚠ ACTIVE SPIKE: +{spike}MB for {obs.spike_steps_remaining} more steps"
            f" — use batch_size=1-2 or EVICT before EXECUTE"
        )
    return base

# ── System prompt — sent once, static content only ───────────────────────────

def get_system_prompt(
    roster_str: str,
    ram_limit_mb: int,
    q_stats: Dict,
    obs: "ModelFlowObservation",
) -> str:
    return f"""You are an expert LLM inference scheduler. Clear the request queue fast while respecting RAM limits.

HARDWARE
  Total RAM    : {ram_limit_mb} MB
  System OS    : {SYSTEM_OVERHEAD_MB} MB always reserved
  effective_free = ram_limit - ram_used - spike_mb - {SYSTEM_OVERHEAD_MB}

{roster_str}

COMMANDS
  LOAD    model_id quant_type                 — load model (check effective_free first)
  EXECUTE model_id quant_type batch_size      — run loaded model on queued requests
  EVICT   model_id quant_type                 — free RAM
  REPLACE model_id quant_type / evict_*       — evict + load in one step (preferred)
  IDLE                                        — do nothing (costs −15, avoid always)

REWARD HINTS
  +50  clear entire queue
  +15  per standard request completed
  +25  per reasoning request completed
  −15  IDLE step
  −30  OOM on LOAD
  −50  OOM on EXECUTE  ← most expensive mistake, always check RAM before EXECUTE
  Efficiency bonus: finish in ≤8 steps

HARD RULES
  1. reasoning requests require Q6_K or Q8_0 — never execute them with Q4/Q5
  2. Before EXECUTE check: ram_used + spike + {SYSTEM_OVERHEAD_MB} + ctx_mb + kv_mb ≤ ram_limit
     If spike is active, use batch_size=1 or wait (EVICT other models to free headroom)
  3. batch_size = min(8, matching_queue_entries) — always maximise throughput
  4. Only EXECUTE a model that has matching queue entries (same model_id)
  5. REPLACE is cheaper than EVICT + LOAD — prefer it when swapping models

OUTPUT: respond with ONE JSON object, no markdown:
{{"command":"LOAD|EXECUTE|EVICT|REPLACE|IDLE","model_id":"...","quant_type":"...","batch_size":8,"evict_model_id":null,"evict_quant_type":null}}"""


# ── Per-step observation — only dynamic state ────────────────────────────────

def observation_to_text(obs: "ModelFlowObservation", q_stats: Dict) -> str:
    feedback = ""
    if obs.last_action_feedback:
        tag = "ERROR" if obs.last_action_error else "OK"
        feedback = f"\nLAST: [{tag}] {obs.last_action_feedback}"

    return (
        f"=== STEP {obs.step_count} | done={obs.info.get('completed',0)} "
        f"pending={obs.info.get('pending',0)} "
        f"cumR={obs.info.get('cumulative_reward',0.0):.1f} ===\n"
        f"{_ram_line(obs)}\n"
        f"{_loaded_summary(obs)}\n"
        f"{_queue_summary(obs, q_stats)}"
        f"{feedback}\n\n"
        f"Next action JSON:"
    )


# ── Message list ─────────────────────────────────────────────────────────────

def build_messages(
    system_prompt: str,
    compressed_history: List[str],
    obs_text: str,
) -> List[Dict]:
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    if compressed_history:
        # Inject history as a single user turn — no fake assistant ACK
        # (the ACK wastes tokens and the LLM doesn't need it)
        history_block = "Step history (oldest→newest):\n" + "\n".join(compressed_history)
        messages.append({"role": "user", "content": history_block})
        messages.append({
            "role": "assistant",
            "content": "Understood — I will avoid repeating past errors."
        })

    messages.append({"role": "user", "content": obs_text})
    return messages