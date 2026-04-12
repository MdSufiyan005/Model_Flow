from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models import ModelFlowObservation
    from inference import DecisionMemory

from config import (
    SYSTEM_OVERHEAD_MB,
    REASONING_MIN_QUANT,
    REASONING_QUANTS,
    RAM_SAFETY_BUFFER_MB,
)

_TIER_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "risky": 2}
_COMPLEXITY_MIN_RANK: Dict[str, int] = {"standard": 0, "reasoning": 2}

EPISODE_LOG_PATH       = Path(os.getenv("EPISODE_LOG_PATH", "episode_log.jsonl"))
_EPISODE_LESSONS_COUNT = 3

EXEC_SAFETY_BUFFER_MB = 2000


# ---------------------------------------------------------------------------
# Cross-episode lesson loader — FIX: filtered by task_name
# ---------------------------------------------------------------------------

def load_past_lessons(n: int = _EPISODE_LESSONS_COUNT, current_task: str = "") -> str:
    """
    Load lessons from past episodes, filtered to the current task only.
    Cross-task lessons caused the agent to apply quality-limit quant rules
    to ram-pressure (which has different RAM constraints and request mixes).
    """
    if not EPISODE_LOG_PATH.exists():
        return ""
    try:
        raw     = EPISODE_LOG_PATH.read_text().strip().splitlines()
        entries = [json.loads(l) for l in raw if l.strip()]

        # Filter to same task if we have enough entries; fall back to all if not.
        if current_task:
            same_task = [e for e in entries if e.get("task", "") == current_task]
            entries = same_task[-n:] if same_task else entries[-n:]
        else:
            entries = entries[-n:]

        if not entries:
            return ""

        lines = ["LESSONS FROM PAST EPISODES (same task only — apply before acting):"]
        for e in entries:
            score_str = f"score={e.get('score', '?'):.2f}" if isinstance(e.get("score"), float) else ""
            lines.append(f"  [{e.get('task','?')}] {score_str} steps={e.get('steps','?')}")
            for err in e.get("top_errors", [])[:2]:
                lines.append(f"    ✗ {err}")
            if ms := e.get("mistake_summary", ""):
                lines.append(f"    → {ms}")
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Roster summary
# ---------------------------------------------------------------------------

def build_roster_str(obs: "ModelFlowObservation") -> str:
    lines = ["ROLE→MODEL ROSTER (size=host_RAM, tps=generation_tokens/s):"]
    for role, info in sorted(obs.model_summary.items()):
        mid = info["model_id"]
        for q, s in sorted(info["stats"].items()):
            cap = "RSN+std" if q in REASONING_QUANTS else "std-only"
            lines.append(
                f"  {role} → {mid}  quant={q}  size={s['size_mb']}MB"
                f"  tier={s['tier']}  tps={s['gen_tps']:.0f}  cap={cap}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision memory block (within-episode)
# ---------------------------------------------------------------------------

def _memory_block(memory: Optional["DecisionMemory"]) -> str:
    if not memory or not memory._entries:
        return ""
    lines = ["WITHIN-EPISODE MEMORY (oldest→newest):"]
    for e in memory._entries[-4:]:
        swap      = f" swap_helped={e.swap_helped}" if e.swap_helped is not None else ""
        band_icon = "✓" if e.band == "good" else ("✗" if e.band == "bad" else "~")
        lines.append(
            f"  [{e.step}] {band_icon} {e.command} {e.model_id or '-'}/{e.quant_type or '-'}"
            f" → {e.band} ({e.result[:50]}){swap}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-model hints
# ---------------------------------------------------------------------------

def _recommended_quant(needs_rsn: bool, quants_available: List[str]) -> str:
    if not quants_available:
        return "Q4_K_M"
    if needs_rsn:
        for q in ["Q6_K", "Q8_0"]:
            if q in quants_available:
                return q
        return sorted(quants_available)[-1]
    else:
        if "Q4_K_M" in quants_available:
            return "Q4_K_M"
        return sorted(quants_available)[0]


def _per_model_hints(obs: "ModelFlowObservation") -> str:
    role_to_model = {r: info["model_id"] for r, info in obs.model_summary.items()}
    free_mb = max(
        0,
        obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb - SYSTEM_OVERHEAD_MB,
    )

    loaded = {slot["model"]: k for k, slot in obs.loaded_models.items()}

    pending: Dict[str, list] = {}
    for req in obs.queue:
        if mid := role_to_model.get(req.model_type):
            pending.setdefault(mid, []).append(req)

    if not pending:
        return "HINTS: no pending requests"

    lines = [
        "HINTS (est_net_gain is approximate — actual reward varies with heat/SLA/quality):"
    ]
    best_model: Optional[str] = None
    best_gain:  float         = -9999.0

    for mid, reqs in sorted(pending.items()):
        needs_rsn = any(r.complexity == "reasoning" for r in reqs)
        oldest    = max(r.age_steps for r in reqs)
        std_c     = sum(1 for r in reqs if r.complexity != "reasoning")
        rsn_c     = len(reqs) - std_c

        quants_available = sorted(
            q for role, info in obs.model_summary.items()
            if info["model_id"] == mid
            for q in info["stats"]
        )
        rec_quant = _recommended_quant(needs_rsn, quants_available)

        size_map: Dict[str, int] = {}
        for info in obs.model_summary.values():
            if info["model_id"] == mid:
                for q, s in info["stats"].items():
                    size_map[q] = s.get("size_mb", 0)

        rec_size_mb = size_map.get(rec_quant, 0)

        rec_tps = 0.0
        for info in obs.model_summary.values():
            if info["model_id"] == mid and rec_quant in info["stats"]:
                rec_tps = info["stats"][rec_quant].get("gen_tps", 0.0)
                break

        rec_post_load_free = free_mb - rec_size_mb
        exec_safe          = rec_post_load_free >= EXEC_SAFETY_BUFFER_MB

        safe_q    = rec_quant
        safe_size = rec_size_mb
        if not exec_safe:
            quality_order = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
            for q in quality_order:
                if q not in quants_available:
                    continue
                sz = size_map.get(q, 0)
                if (free_mb - sz) >= EXEC_SAFETY_BUFFER_MB:
                    safe_q    = q
                    safe_size = sz
                    break

        fits_ram      = rec_size_mb <= free_mb
        fits_ram_safe = safe_size <= free_mb

        servable   = 0
        loaded_as  = "not-loaded"
        loaded_rec = False
        if mid in loaded:
            key  = loaded[mid]
            slot = obs.loaded_models[key]
            tier_r    = _TIER_RANK.get(slot.get("tier", "low"), 0)
            servable  = min(8, sum(
                1 for r in reqs if _COMPLEXITY_MIN_RANK.get(r.complexity, 0) <= tier_r
            ))
            loaded_as  = key
            loaded_rec = key == f"{mid}-{rec_quant}"

        net = min(std_c * 15 + rsn_c * 25, servable * 25) - (len(obs.queue) - servable) * 3

        if net > best_gain:
            best_gain  = net
            best_model = mid

        lines.append(
            f"  {mid}:"
            f" loaded={loaded_as}"
            f" servable={servable}"
            f" oldest_age={oldest}steps"
            f" std={std_c} rsn={rsn_c}"
            f" recommended_quant={rec_quant} (fits_RAM={fits_ram}, size={rec_size_mb}MB,"
            f" tps={rec_tps:.0f}, exec_safe={exec_safe})"
            f" safe_quant={safe_q} (size={safe_size}MB, fits_RAM={fits_ram_safe})"
            f" rec_already_loaded={loaded_rec}"
            f" est_net_gain={net}"
        )

        if not exec_safe and safe_q != rec_quant:
            lines.append(
                f"    ⚠ exec_safe=False for {rec_quant}: post-load headroom only"
                f" {rec_post_load_free:.0f}MB < {EXEC_SAFETY_BUFFER_MB}MB needed."
                f" Use safe_quant={safe_q} to avoid RUNTIME OOM."
                f" Reasoning requests will be served at slightly lower quality — that"
                f" is better than a failed EXECUTE."
            )

    if best_model:
        lines.append(f"BEST TARGET: {best_model} (est_net_gain={best_gain:.0f})")
        if best_gain <= 0:
            lines.append(
                "  ⚠ est_net_gain ≤ 0: requests are stale or wrong model is loaded."
                " Check queue — if the right model is not loaded, LOAD or REPLACE it."
                " Do NOT load a model that is already in LOADED MODELS."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loaded model summary
# ---------------------------------------------------------------------------

def _loaded_summary(obs: "ModelFlowObservation") -> str:
    if not obs.loaded_models:
        return "LOADED: none"
    heat  = getattr(obs, "model_heat_signals", {})
    lines = ["LOADED MODELS:"]
    for key, slot in obs.loaded_models.items():
        h = heat.get(key, "?")
        lines.append(
            f"  {key}  tier={slot['tier']}  size={slot['size_mb']}MB"
            f"  gen_tps={slot['gen_tps']:.0f}  heat={h}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal summary
# ---------------------------------------------------------------------------

def _signal_summary(obs: "ModelFlowObservation") -> str:
    parts = []
    if q := getattr(obs, "recent_quality_outcomes", None):
        symbol = "".join("✓" if x else "✗" for x in q[-3:])
        parts.append(f"RECENT_QUALITY={symbol or 'none'}")
    if getattr(obs, "demand_hint", None) == "shift_detected":
        parts.append("DEMAND=SHIFTED⚠ (queue type mix changed — re-check HINTS)")
    if sla := getattr(obs, "current_sla_steps", None):
        parts.append(f"SLA_WINDOW={sla}steps")
    if (defc := getattr(obs, "deferred_count", 0)) > 0:
        parts.append(f"DEFERRED={defc}")
    return "SIGNALS: " + (" | ".join(parts) if parts else "none")


# ---------------------------------------------------------------------------
# Queue summary — FIX: added REMAINING breakdown per model
# ---------------------------------------------------------------------------

def _queue_summary(obs: "ModelFlowObservation", q_stats: Dict) -> str:
    if not obs.queue:
        return "QUEUE: empty"

    role_to_model = {r: info["model_id"] for r, info in obs.model_summary.items()}

    # FIX: explicit per-model remaining count so agent never loses track
    remaining_by_model: Dict[str, int] = {}
    for req in obs.queue:
        mid = role_to_model.get(req.model_type, req.model_type)
        remaining_by_model[mid] = remaining_by_model.get(mid, 0) + 1

    lines = [f"QUEUE: {len(obs.queue)} pending"]
    lines.append(
        "REMAINING: "
        + (", ".join(f"{m}={c}" for m, c in sorted(remaining_by_model.items()))
           or "none")
    )

    for mid, st in q_stats.items():
        s = st.get("standard", 0)
        r = st.get("reasoning", 0)
        if s + r > 0:
            lines.append(f"  {mid}: {s} standard + {r} reasoning")
    lines.append("NEXT 4:")
    for req in obs.queue[:4]:
        lines.append(
            f"  [{req.request_id}] {req.model_type} {req.complexity} age={req.age_steps}steps"
        )
    if len(obs.queue) > 4:
        lines.append(f"  ... +{len(obs.queue) - 4} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAM status line
# ---------------------------------------------------------------------------

def _ram_line(obs: "ModelFlowObservation") -> str:
    free = max(
        0,
        obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb - SYSTEM_OVERHEAD_MB,
    )
    s = f"RAM: used={obs.ram_used_mb}MB / limit={obs.ram_limit_mb}MB / usable_free={free}MB"
    if obs.pressure_spike_mb > 0:
        s += (
            f"  ⚠ SPIKE +{obs.pressure_spike_mb}MB"
            f" ({obs.spike_steps_remaining} steps left)"
            f" → batch_size ≤ 2"
        )
    return s


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------

def _loop_detector(memory: Optional["DecisionMemory"], obs) -> str:
    if not memory or not memory._entries:
        return ""

    recent = memory._entries

    if len(recent) >= 1:
        last = recent[-1]
        if last.command == "REPLACE" and last.band == "bad":
            loaded_keys = set(obs.loaded_models.keys())
            if last.model_id and last.quant_type:
                target_key = f"{last.model_id}-{last.quant_type}"
                if target_key in loaded_keys:
                    return (
                        f"🔁 REPLACE SELF-LOOP: last REPLACE targeted {target_key}"
                        f" which is ALREADY LOADED. Do NOT REPLACE again."
                        f" → EXECUTE {target_key} immediately (batch_size=8)."
                    )
            return (
                f"🔁 BAD REPLACE DETECTED: last REPLACE lost reward."
                f" Check LOADED MODELS — if target is already loaded, use EXECUTE."
                f" If target was evicted accidentally, use LOAD (not REPLACE)."
            )

    recent_bad = sum(1 for e in recent[-2:] if e.band == "bad")
    if recent_bad >= 2:
        bad_cmds = [e.command for e in recent[-2:] if e.band == "bad"]
        return (
            f"🔁 LOOP DETECTED: last 2 actions bad ({', '.join(bad_cmds)})."
            f" If last error was RUNTIME OOM → follow RUNTIME OOM RECOVERY in system prompt."
            f" If last error was 'already loaded' → EXECUTE (not LOAD/REPLACE)."
            f" Otherwise restart from STEP 1 of DECISION TREE."
        )

    return ""


# ---------------------------------------------------------------------------
# System prompt — FIX: accepts current_task for lesson filtering
# ---------------------------------------------------------------------------

def get_system_prompt(
    roster_str: str,
    ram_limit_mb: int,
    q_stats: Dict,
    obs: "ModelFlowObservation",
    current_task: str = "",
) -> str:
    past_lessons  = load_past_lessons(current_task=current_task)
    lessons_block = f"\n{past_lessons}\n" if past_lessons else ""

    return f"""You are an expert LLM inference scheduler. Goal: clear the request queue fast while respecting RAM and quality constraints.

HARDWARE: {ram_limit_mb}MB total. {SYSTEM_OVERHEAD_MB}MB reserved for OS.

{roster_str}
{lessons_block}
QUANT SELECTION RULE (critical — this is where most mistakes happen):
  • Queue has ANY reasoning request for a model → use Q6_K (or Q8_0 if Q6_K unavailable)
  • Queue has ONLY standard requests for a model → use Q4_K_M
  • Using Q6_K on standard-only requests wastes RAM and incurs a small penalty
  • The HINTS block shows recommended_quant — but ALWAYS check exec_safe first (see below)
  • IMPORTANT: check per-model, not globally. One model needing Q6_K does NOT mean all models need Q6_K.

RUNTIME OOM RECOVERY (check this BEFORE the decision tree if last error was RUNTIME OOM):
  RUNTIME OOM means the loaded quant's execution footprint exceeds available RAM.
  Recovery steps:
    1. EVICT the offending model immediately.
    2. Look at HINTS → find safe_quant for that model (exec_safe=True).
    3. LOAD safe_quant instead (usually Q4_K_M). It uses less KV-cache RAM.
    4. EXECUTE with safe_quant. Reasoning quality drops slightly but execution succeeds.
  NEVER retry EXECUTE with the same quant after a RUNTIME OOM — it will OOM again.

EXEC SAFETY RULE (prevents RUNTIME OOM before it happens):
  Before loading any quant, check HINTS → exec_safe field for that model.
  • exec_safe=True  → recommended_quant is safe to load AND execute.
  • exec_safe=False → recommended_quant will likely cause RUNTIME OOM.
                      Use safe_quant from HINTS instead.

QUEUE ACCOUNTING RULE (prevents no-match EXECUTE loops):
  The QUEUE block now shows REMAINING: model=count for every model with pending work.
  Before issuing ANY action, read REMAINING carefully:
  • If a model shows count=0 or does not appear → it has NO pending requests.
    Do NOT EXECUTE or REPLACE to load that model — it will waste steps.
  • Only LOAD/REPLACE/EXECUTE models that appear in REMAINING with count > 0.
  • After each EXECUTE, mentally subtract the batch served from REMAINING.

ALREADY-LOADED RULE (prevents REPLACE and LOAD self-loops):
  Before issuing LOAD or REPLACE, always check LOADED MODELS.
  • If the model+quant you want is already in LOADED MODELS → EXECUTE it directly.
  • NEVER issue REPLACE where model_id+quant_type matches an already-loaded key.
  • NEVER issue REPLACE with evict_model_id+evict_quant_type == model_id+quant_type.

DECISION TREE — follow in order every turn:

STEP 1: Read REMAINING. Identify which models still have pending requests.
  → If REMAINING is empty or shows all zeros → queue is done, episode should end.

STEP 2: Look at HINTS → find model where rec_already_loaded=True AND servable > 0
  → Found: go to STEP 3
  → Not found: go to STEP 4

STEP 3: EXECUTE
  Use: model from STEP 2, exact quant already loaded, batch_size=min(8, servable)
  If SPIKE active: batch_size=min(2, servable)
  → STOP

STEP 4: Load or replace the right model
  From HINTS, pick BEST TARGET (must appear in REMAINING with count > 0).
  Determine which quant to use:
    a) exec_safe=True  → use recommended_quant
    b) exec_safe=False → use safe_quant (shown in HINTS)
  Then:
    i)  fits_RAM=True AND model not loaded → LOAD with chosen quant → STOP
    ii) fits_RAM=False OR need to free space → REPLACE:
          evict_model_id = model with count=0 in REMAINING (least needed)
          model_id = BEST TARGET, quant_type = chosen quant → STOP
  ⚠ CRITICAL: before issuing REPLACE, verify model_id+quant_type is NOT in LOADED MODELS.
              If it is → go to STEP 3 instead.

STEP 5: Nothing actionable → IDLE (costs -15, last resort only)

ABSOLUTE CONSTRAINTS:
  • Never LOAD a model+quant that appears in LOADED MODELS — costs -5 and wastes a step
  • Never EXECUTE a model+quant not in LOADED MODELS — costs -10
  • Never EXECUTE a model that has count=0 in REMAINING — wastes steps
  • batch_size ≤ 8 always
  • For REPLACE: always set evict_model_id AND evict_quant_type explicitly
  • Prefer to evict models whose count=0 in REMAINING (no pending work)
  • Never retry EXECUTE after RUNTIME OOM with the same quant
  • Never REPLACE a model with itself

Output ONE JSON object, no text before or after:
{{"command":"LOAD|EXECUTE|EVICT|REPLACE|DEFER|IDLE","model_id":"...","quant_type":"...","batch_size":N,"evict_model_id":null,"evict_quant_type":null}}"""


# ---------------------------------------------------------------------------
# Observation text — FIX: passes current_task to get_system_prompt
# ---------------------------------------------------------------------------

def observation_to_text(
    obs: "ModelFlowObservation",
    q_stats: Dict,
    memory: Optional["DecisionMemory"] = None,
    last_was_overridden: bool = False,
    overridden_from: Optional[str] = None,
    overridden_to: Optional[str] = None,
) -> str:
    sections: List[str] = []

    if last_was_overridden:
        sections.append(
            f"⚠️  LAST ACTION INVALID — NOT EXECUTED.\n"
            f"   Proposed: {overridden_from or 'unknown'}\n"
            f"   See LAST ACTION below for reason. Choose a different action."
        )

    sections.append(
        f"=== STEP {obs.step_count}"
        f" | completed={obs.info.get('completed', 0)}"
        f" | pending={obs.info.get('pending', 0)}"
        f" | deferred={obs.info.get('deferred', 0)}"
        f" | cumR={obs.info.get('cumulative_reward', 0):.1f} ==="
    )

    sections.append(_ram_line(obs))
    sections.append(_loaded_summary(obs))
    sections.append(_signal_summary(obs))

    mem_block = _memory_block(memory)
    if mem_block:
        sections.append(mem_block)

    loop_warning = _loop_detector(memory, obs)
    if loop_warning:
        sections.append(loop_warning)

    sections.append(_per_model_hints(obs))
    sections.append(_queue_summary(obs, q_stats))

    if obs.last_action_feedback:
        status = "ERROR" if obs.last_action_error else "OK"
        sections.append(f"LAST ACTION: {status} — {obs.last_action_feedback[:200]}")

    sections.append("\nApply DECISION TREE. Output JSON only:")

    return "\n".join(filter(None, sections))


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, obs_text: str) -> List[Dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": obs_text},
    ]