"""
prompt.py — ModelFlow V2

Changes in this version
------------------------
1. Quant selection made EXPLICIT in hints:
   Previous version said "best_quant=Q4_K_M" but the decision tree still
   said "reasoning needs Q6_K or higher" without specifying when to use Q4_K_M.
   The LLM resolved the ambiguity by always using Q6_K "to be safe".
   Fix: the hints block now states the EXACT recommended quant per model
   based on the actual queue mix for that model, and the decision tree
   explicitly says "use Q4_K_M for standard-only queues".

2. gen_tps and host_mb exposed in hints so the LLM can reason about
   the speed/RAM tradeoff. The tick cap (MAX_EXEC_TICKS=8) means the
   agent no longer needs to fear slow models — but it should still prefer
   faster ones when the queue is large and aging.

3. Recommended_quant field added to hints: a single string the LLM should
   copy directly into its JSON output. This eliminates the ambiguity of
   "best_quant" (which was advisory) vs "required_quant" (hard rule).

4. System prompt decision tree updated: Step 3 now explicitly says which
   quant to use based on queue complexity, with Q4_K_M as the default.

5. Cross-episode lessons, override pre-warning, loop detector, memory
   block all preserved from previous version.
"""

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


# ---------------------------------------------------------------------------
# Cross-episode lesson loader
# ---------------------------------------------------------------------------

def load_past_lessons(n: int = _EPISODE_LESSONS_COUNT) -> str:
    if not EPISODE_LOG_PATH.exists():
        return ""
    try:
        raw     = EPISODE_LOG_PATH.read_text().strip().splitlines()
        entries = [json.loads(l) for l in raw[-n:] if l.strip()]
        if not entries:
            return ""
        lines = ["LESSONS FROM PAST EPISODES (apply before acting):"]
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
# Per-model hints — now includes recommended_quant and gen_tps
# ---------------------------------------------------------------------------

def _recommended_quant(needs_rsn: bool, quants_available: List[str]) -> str:
    """
    Return the single best quant to use for this model given the request mix.

    Rule:
      - Any reasoning request present → need tier=high → use REASONING_MIN_QUANT
        (Q6_K) or higher if available.
      - Standard only → use Q4_K_M (or the lowest available quant that is
        at least medium tier, i.e. Q4_K_M). Never waste Q6_K on std-only.

    The returned string is the EXACT value to put in quant_type JSON field.
    """
    if not quants_available:
        return "Q4_K_M"

    if needs_rsn:
        # Prefer Q6_K; fall back to highest available that qualifies.
        for q in ["Q6_K", "Q8_0"]:
            if q in quants_available:
                return q
        # No Q6_K available — return highest available and flag the issue.
        return sorted(quants_available)[-1]
    else:
        # Standard only: use Q4_K_M. If not available, use lowest available.
        if "Q4_K_M" in quants_available:
            return "Q4_K_M"
        return sorted(quants_available)[0]


def _per_model_hints(obs: "ModelFlowObservation") -> str:
    """
    Scheduling hints per model with pending requests.

    Key fields:
      recommended_quant  — exact quant_type string to use in your JSON
      servable           — how many requests the currently loaded slot can serve
      est_net_gain       — rough estimate (actual reward differs by heat/SLA/quality)
      gen_tps            — generation speed; faster = fewer internal penalty ticks
    """
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
    best_model:     Optional[str] = None
    best_gain:      float         = -9999.0

    for mid, reqs in sorted(pending.items()):
        needs_rsn = any(r.complexity == "reasoning" for r in reqs)
        oldest    = max(r.age_steps for r in reqs)
        std_c     = sum(1 for r in reqs if r.complexity != "reasoning")
        rsn_c     = len(reqs) - std_c

        # Gather quants available for this model from the roster.
        quants_available = sorted(
            q for role, info in obs.model_summary.items()
            if info["model_id"] == mid
            for q in info["stats"]
        )
        rec_quant = _recommended_quant(needs_rsn, quants_available)

        # gen_tps for the recommended quant (so LLM knows speed).
        rec_tps = 0.0
        for role, info in obs.model_summary.items():
            if info["model_id"] == mid and rec_quant in info["stats"]:
                rec_tps = info["stats"][rec_quant].get("gen_tps", 0.0)
                break

        # Size of recommended quant (so LLM can check RAM).
        rec_size_mb = 0
        for role, info in obs.model_summary.items():
            if info["model_id"] == mid and rec_quant in info["stats"]:
                rec_size_mb = info["stats"][rec_quant].get("size_mb", 0)
                break

        # Servable: how many current-queue requests the loaded slot can handle.
        servable   = 0
        loaded_as  = "not-loaded"
        loaded_rec = False   # is the recommended quant already loaded?
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

        fits_ram = rec_size_mb <= free_mb

        lines.append(
            f"  {mid}:"
            f" loaded={loaded_as}"
            f" servable={servable}"
            f" oldest_age={oldest}steps"
            f" std={std_c} rsn={rsn_c}"
            f" recommended_quant={rec_quant} (fits_RAM={fits_ram}, size={rec_size_mb}MB, tps={rec_tps:.0f})"
            f" rec_already_loaded={loaded_rec}"
            f" est_net_gain={net}"
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
# Loaded model summary (with heat and tps)
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
# V2 signal summary
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
# Queue summary
# ---------------------------------------------------------------------------

def _queue_summary(obs: "ModelFlowObservation", q_stats: Dict) -> str:
    if not obs.queue:
        return "QUEUE: empty"
    lines = [f"QUEUE: {len(obs.queue)} pending"]
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
# System prompt
# ---------------------------------------------------------------------------

def get_system_prompt(
    roster_str: str,
    ram_limit_mb: int,
    q_stats: Dict,
    obs: "ModelFlowObservation",
) -> str:
    past_lessons  = load_past_lessons()
    lessons_block = f"\n{past_lessons}\n" if past_lessons else ""

    return f"""You are an expert LLM inference scheduler. Goal: clear the request queue fast while respecting RAM and quality constraints.

HARDWARE: {ram_limit_mb}MB total. {SYSTEM_OVERHEAD_MB}MB reserved for OS.

{roster_str}
{lessons_block}
QUANT SELECTION RULE (critical — this is where most mistakes happen):
  • Queue has ANY reasoning request for a model → use Q6_K (or Q8_0 if Q6_K unavailable)
  • Queue has ONLY standard requests for a model → use Q4_K_M
  • Using Q6_K on standard-only requests wastes RAM and incurs a small penalty
  • The HINTS block shows recommended_quant — copy it exactly into your JSON

DECISION TREE — follow in order every turn:

STEP 1: Look at HINTS → find model where rec_already_loaded=True AND servable > 0
  → Found: go to STEP 2
  → Not found: go to STEP 3

STEP 2: EXECUTE
  Use: model from STEP 1, exact quant already loaded, batch_size=min(8, servable)
  If SPIKE active: batch_size=min(2, servable)
  → STOP

STEP 3: Load or replace the right model
  From HINTS, pick BEST TARGET. Use recommended_quant (exact string from hints).
  a) Is recommended_quant already loaded? → it would show rec_already_loaded=True → already handled in STEP 1
  b) fits_RAM=True AND model not loaded → LOAD it with recommended_quant → STOP
  c) fits_RAM=False OR need to free space → REPLACE:
       evict_model_id = least-needed loaded model (not needed by any queue request)
       model_id = BEST TARGET, quant_type = recommended_quant → STOP

STEP 4: Nothing actionable → IDLE (costs -15, last resort only)

ABSOLUTE CONSTRAINTS:
  • Never LOAD a model+quant that appears in LOADED MODELS — costs -5 and wastes a step
  • Never EXECUTE a model+quant not in LOADED MODELS — costs -10
  • batch_size ≤ 8 always
  • For REPLACE: always set evict_model_id AND evict_quant_type explicitly

Output ONE JSON object, no text before or after:
{{"command":"LOAD|EXECUTE|EVICT|REPLACE|DEFER|IDLE","model_id":"...","quant_type":"...","batch_size":N,"evict_model_id":null,"evict_quant_type":null}}"""


# ---------------------------------------------------------------------------
# Observation text
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

    # Override pre-warning at top — LLM sees it before it reasons.
    if last_was_overridden:
        sections.append(
            f"⚠️  LAST ACTION INVALID — NOT EXECUTED.\n"
            f"   Proposed: {overridden_from or 'unknown'}\n"
            f"   See LAST ACTION below for reason. Choose a different action."
        )

    # Step header.
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

    # Loop detector.
    if memory:
        recent_bad = sum(1 for e in memory._entries[-2:] if e.band == "bad")
        if recent_bad >= 2:
            bad_cmds = [e.command for e in memory._entries[-2:] if e.band == "bad"]
            sections.append(
                f"🔁 LOOP: last 2 actions bad ({', '.join(bad_cmds)})."
                f" Restart from STEP 1 of DECISION TREE."
            )

    sections.append(_per_model_hints(obs))
    sections.append(_queue_summary(obs, q_stats))

    if obs.last_action_feedback:
        status = "ERROR" if obs.last_action_error else "OK"
        # Show full feedback so agent sees ticks_used/raw and contention.
        sections.append(f"LAST ACTION: {status} — {obs.last_action_feedback[:200]}")

    sections.append("\nApply DECISION TREE. Output JSON only:")

    return "\n".join(filter(None, sections))


def build_messages(system_prompt: str, obs_text: str) -> List[Dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": obs_text},
    ]