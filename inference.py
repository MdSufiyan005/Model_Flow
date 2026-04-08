import os
import json
import time
import sys
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
from models import ModelFlowAction, ModelFlowObservation
from server.modelflow_environment import ModelFlowEnvironment

load_dotenv()

def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

USE_GROQ_ONLY = env_bool("USE_GROQ_ONLY")

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

if USE_GROQ_ONLY:
    if not GROQ_API_KEY:
        raise ValueError("USE_GROQ_ONLY=1 but GROQ_API_KEY is missing")
    from groq import Groq
    client       = Groq(api_key=GROQ_API_KEY)
    active_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    # print("groq", file=sys.stderr)
else:
    if not API_KEY:
        raise ValueError("OpenAI/HF branch selected but HF_TOKEN/API_KEY is missing")
    from openai import OpenAI
    client       = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    active_model = MODEL_NAME

# Config 

BENCHMARK             = "modelflow"
TASKS                 = ["single-load", "multi-load", "quality-limit", "ram-pressure"]
MAX_STEPS_PER_TASK    = 30
TEMPERATURE           = 0.1
MAX_TOKENS            = 600
CONTEXT_HISTORY_STEPS = 6
MAX_RETRIES           = 4
BASE_BACKOFF_S        = 2.0
SYSTEM_OVERHEAD_MB    = 1100

ROLE_TO_MODEL: Dict[str, str] = {
    "chatbot":    "gemma-3-4b",
    "translator": "llama_1b",
    "coder":      "qwen3.5-2b",
}

# Quant tiers in ascending size order
QUANT_TIER: Dict[str, str] = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K":   "high",
    "Q8_0":   "risky",
}

# Minimum tier required to serve reasoning
REASONING_MIN_QUANT = "Q6_K"
REASONING_QUANTS    = {"Q6_K", "Q8_0"}

# : RAM budget calculator 

def ram_free(obs: ModelFlowObservation) -> int:
    """
    Effective free RAM accounting for system overhead and any active spike.
    Uses the info dict when available, falls back to observation fields.
    """
    return int(obs.info.get(
        "ram_free_mb",
        obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb
    ))


def model_host_mb(obs: ModelFlowObservation, model_id: str, quant_type: str) -> int:
    """Return the host_mb for a given model+quant from model_summary roster stats."""
    for role, info in obs.model_summary.items():
        if info["model_id"] == model_id:
            stats = info["stats"].get(quant_type)
            if stats:
                return stats["size_mb"]
    return 99999  # unknown → treat as too large


def can_load(obs: ModelFlowObservation, model_id: str, quant_type: str) -> bool:
    """True if loading model_id-quant_type fits within effective free RAM."""
    needed = model_host_mb(obs, model_id, quant_type)
    free   = ram_free(obs)
    return needed <= free


# Queue intelligence helpers 

def queue_stats(obs: ModelFlowObservation) -> Dict[str, Dict]:
    """
    Per-model breakdown of pending requests.
    Returns:
        { model_id: { "total": int, "reasoning": int, "standard": int } }
    """
    stats: Dict[str, Dict] = {}
    for req in obs.queue:
        mid = ROLE_TO_MODEL.get(req.model_type, req.model_type)
        if mid not in stats:
            stats[mid] = {"total": 0, "reasoning": 0, "standard": 0}
        stats[mid]["total"] += 1
        if req.complexity == "reasoning":
            stats[mid]["reasoning"] += 1
        else:
            stats[mid]["standard"] += 1
    return stats


def required_quant(model_id: str, obs: ModelFlowObservation) -> str:
    """
    Choose the minimum sufficient quant for model_id given queue contents.
    If ANY reasoning request exists for this model → Q6_K.
    Otherwise → Q4_K_M (cheapest tier that handles standard).
    """
    qs = queue_stats(obs)
    info = qs.get(model_id, {})
    if info.get("reasoning", 0) > 0:
        return REASONING_MIN_QUANT
    return "Q4_K_M"


def loaded_key(obs: ModelFlowObservation, model_id: str) -> Optional[str]:
    """Return the loaded dict key for model_id if it is currently resident, else None."""
    for key in obs.loaded_models:
        if obs.loaded_models[key]["model"] == model_id:
            return key
    return None


def loaded_quant(obs: ModelFlowObservation, model_id: str) -> Optional[str]:
    """Return the quant_type currently loaded for model_id, or None."""
    key = loaded_key(obs, model_id)
    if key:
        return obs.loaded_models[key]["quant"]
    return None


def can_serve_reasoning(obs: ModelFlowObservation, model_id: str) -> bool:
    """True if the currently loaded quant for model_id can serve reasoning."""
    q = loaded_quant(obs, model_id)
    return q in REASONING_QUANTS if q else False


_KNOWN_MODELS = {"gemma-3-4b", "llama_1b", "qwen3.5-2b"}
_KNOWN_QUANTS = {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}


def _normalise_model_id(raw_model: Optional[str], raw_quant: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle the hallucination where the LLM concatenates quant into model_id,
    e.g. model_id="qwen3.5-2b-Q6_K", quant_type=null.
    Also strip any trailing whitespace/quotes.
    """
    if not raw_model:
        return raw_model, raw_quant

    model = raw_model.strip().strip('"').strip("'")

    # If quant is missing and the model string contains a known quant suffix, split them
    if not raw_quant:
        for q in _KNOWN_QUANTS:
            if model.endswith(f"-{q}"):
                return model[: -(len(q) + 1)], q

    # Validate model name
    if model not in _KNOWN_MODELS:
        # Try prefix match (e.g. "gemma" → "gemma-3-4b")
        for m in _KNOWN_MODELS:
            if m.startswith(model) or model.startswith(m.split("-")[0]):
                model = m
                break

    quant = raw_quant.strip().strip('"').strip("'") if raw_quant else raw_quant
    if quant and quant not in _KNOWN_QUANTS:
        quant = None  # unknown quant → let planner decide

    return model, quant


def parse_action(response_text: str) -> Dict:
    try:
        start = response_text.find("{")
        end   = response_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        data    = json.loads(response_text[start:end])
        command = str(data.get("command", "IDLE")).upper().strip()
        if command not in {"LOAD", "EXECUTE", "EVICT", "IDLE", "REPLACE"}:
            command = "IDLE"

        raw_model      = data.get("model_id")
        raw_quant      = data.get("quant_type")
        raw_evict_m    = data.get("evict_model_id")
        raw_evict_q    = data.get("evict_quant_type")

        model_id,   quant_type   = _normalise_model_id(raw_model,   raw_quant)
        evict_model, evict_quant = _normalise_model_id(raw_evict_m, raw_evict_q)

        batch_size = data.get("batch_size")
        if batch_size is None or not isinstance(batch_size, (int, float)):
            batch_size = 4
        batch_size = max(1, min(int(batch_size), 8))

        return {
            "command":          command,
            "model_id":         model_id,
            "quant_type":       quant_type,
            "batch_size":       batch_size,
            "evict_model_id":   evict_model,
            "evict_quant_type": evict_quant,
        }
    except Exception as e:
        # print(f"[PARSE ERROR]: {e}", file=sys.stderr)
        return {"command": "IDLE", "model_id": None, "quant_type": None,
                "batch_size": 1, "evict_model_id": None, "evict_quant_type": None}


# Pre-action safety override 

def get_eviction_target(obs: ModelFlowObservation, exclude_model_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Returns (model_id, quant_type) of a loaded model to evict.
    Prefers largest non-needed model. If all are needed, falls back to largest other model."""
    q_stats = queue_stats(obs)
    candidates = []
    for key, slot in obs.loaded_models.items():
        mid = slot["model"]
        if mid == exclude_model_id:
            continue
        pending = q_stats.get(mid, {}).get("total", 0)
        candidates.append({
            "model_id": mid,
            "quant_type": slot["quant"],
            "size": slot.get("size_mb", 0),
            "needed": pending > 0
        })
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: (c["needed"], -c["size"]))
    best = candidates[0]
    return best["model_id"], best["quant_type"]


def apply_planning_override(
    action: ModelFlowAction,
    obs:    ModelFlowObservation,
) -> ModelFlowAction:
    """
    Intercept and correct actions before they reach the environment.

    Fixes applied:
      1. LOAD/REPLACE RAM pre-flight check — prevent OOM before it happens.
      2. Quant upgrade on LOAD — if reasoning requests exist, enforce Q6_K.
      3. Quant upgrade on REPLACE — same rule.
      4. Block EXECUTE on empty slot — don't execute if no matching requests remain.
      5. Block duplicate LOAD — don't load a model that's already resident.
      6. Suggest EVICT before LOAD when RAM is too tight.
    """
    q_stats = queue_stats(obs)

    # LOAD override 
    if action.command == "LOAD":
        if not action.model_id or not action.quant_type:
            action.command = "IDLE"
            return action

        # 5. Already loaded?
        if loaded_key(obs, action.model_id):
            # It's there but might be the wrong quant — convert to REPLACE
            current_quant = loaded_quant(obs, action.model_id)
            needed_quant  = required_quant(action.model_id, obs)
            if current_quant != needed_quant:
                action.command         = "REPLACE"
                action.evict_model_id  = action.model_id
                action.evict_quant_type = current_quant
                action.quant_type      = needed_quant
            else:
                action.command = "IDLE"
            return action

        # 2. Quant upgrade if reasoning pending
        needed_quant = required_quant(action.model_id, obs)
        if action.quant_type not in REASONING_QUANTS and needed_quant in REASONING_QUANTS:
            action.quant_type = needed_quant

        # 1. RAM pre-flight
        if not can_load(obs, action.model_id, action.quant_type):
            # Try one tier lower before giving up (e.g. Q6_K → Q5_K_M)
            # but only if no reasoning requests need this model
            model_qs   = q_stats.get(action.model_id, {})
            has_reason = model_qs.get("reasoning", 0) > 0
            fallback   = {"Q8_0": "Q6_K", "Q6_K": "Q5_K_M", "Q5_K_M": "Q4_K_M"}
            fb_quant   = fallback.get(action.quant_type)
            if fb_quant and not has_reason and can_load(obs, action.model_id, fb_quant):
                action.quant_type = fb_quant
            else:
                evict_m, evict_q = get_eviction_target(obs, exclude_model_id=action.model_id)
                if evict_m:
                    # print(f"[OVERRIDE] LOAD({action.model_id}-{action.quant_type}) blocked: "
                    #       f"needs {model_host_mb(obs, action.model_id, action.quant_type)}MB, "
                    #       f"free={ram_free(obs)}MB. Converting to EVICT({evict_m}-{evict_q}).", file=sys.stderr)
                    action.command = "EVICT"
                    action.model_id = evict_m
                    action.quant_type = evict_q
                    action.batch_size = 0
                else:
                    # print(f"[OVERRIDE] LOAD({action.model_id}-{action.quant_type}) blocked: "
                    #       f"needs {model_host_mb(obs, action.model_id, action.quant_type)}MB, "
                    #       f"free={ram_free(obs)}MB. Emitting IDLE.", file=sys.stderr)
                    action.command = "IDLE"

    #REPLACE override 
    elif action.command == "REPLACE":
        if not action.model_id or not action.quant_type:
            action.command = "IDLE"
            return action

        # 2. Quant upgrade for reasoning
        needed_quant = required_quant(action.model_id, obs)
        if action.quant_type not in REASONING_QUANTS and needed_quant in REASONING_QUANTS:
            action.quant_type = needed_quant

        # Infer evict target if not specified
        if not action.evict_model_id and not action.evict_quant_type:
            # Default: evict the same model (quant upgrade in place)
            action.evict_model_id   = action.model_id
            action.evict_quant_type = loaded_quant(obs, action.model_id)

        # 1. RAM pre-flight (after simulating the eviction)
        evict_size = 0
        if action.evict_model_id and action.evict_quant_type:
            evict_key_str = f"{action.evict_model_id}-{action.evict_quant_type}"
            if evict_key_str in obs.loaded_models:
                evict_size = obs.loaded_models[evict_key_str].get("size_mb", 0)

        simulated_free = ram_free(obs) + evict_size
        needed_mb      = model_host_mb(obs, action.model_id, action.quant_type)
        if needed_mb > simulated_free:
            evict_m, evict_q = get_eviction_target(obs, exclude_model_id=action.model_id)
            if evict_m:
                # print(f"[OVERRIDE] REPLACE({action.model_id}-{action.quant_type}) blocked: "
                #       f"needs {needed_mb}MB, simulated_free={simulated_free}MB. Converting to EVICT({evict_m}-{evict_q}).", file=sys.stderr)
                action.command = "EVICT"
                action.model_id = evict_m
                action.quant_type = evict_q
                action.evict_model_id = None
                action.evict_quant_type = None
                action.batch_size = 0
            else:
                # print(f"[OVERRIDE] REPLACE({action.model_id}-{action.quant_type}) blocked: "
                #       f"needs {needed_mb}MB, simulated_free={simulated_free}MB. Emitting IDLE.", file=sys.stderr)
                action.command = "IDLE"

    # EXECUTE override 
    elif action.command == "EXECUTE":
        if not action.model_id or not action.quant_type:
            # Try to fill from the only loaded model
            if len(obs.loaded_models) == 1:
                only = list(obs.loaded_models.values())[0]
                action.model_id   = only["model"]
                action.quant_type = only["quant"]
            else:
                action.command = "IDLE"
                return action

        # 5. Nothing to execute for this model?
        model_qs = q_stats.get(action.model_id, {})
        if model_qs.get("total", 0) == 0:
            # print(f"[OVERRIDE] EXECUTE({action.model_id}) blocked: no pending requests.", file=sys.stderr)
            action.command = "IDLE"
            return action
        
        # 4. Tier check — don't let medium quant attempt reasoning again
        if model_qs.get("reasoning", 0) > 0 and not can_serve_reasoning(obs, action.model_id):
            # print(f"[OVERRIDE] EXECUTE({action.model_id}-{action.quant_type}) blocked: "
            #       f"reasoning pending but tier too low.", file=sys.stderr)
            action.command = "IDLE"
            return action

        # Cap batch to actual pending count for this model
        pending_total = model_qs.get("total", 8)
        action.batch_size = min(action.batch_size, pending_total, 8)

        # Check peak RAM against budget
        exec_loaded_key = f"{action.model_id}-{action.quant_type}"
        slot = obs.loaded_models.get(exec_loaded_key)
        if slot:
            ctx = slot.get("ctx_mb", 0)
            comp = slot.get("comp_mb", 0)
            kv_max = slot.get("kv_mb_max", 0)
            kv_dyn = kv_max * (action.batch_size / 8.0)
            peak_dyn = ctx + comp + kv_dyn
            total_peak = obs.ram_used_mb + obs.pressure_spike_mb + SYSTEM_OVERHEAD_MB + peak_dyn
            if total_peak > obs.ram_limit_mb:
                evict_m, evict_q = get_eviction_target(obs, exclude_model_id=action.model_id)
                if evict_m:
                    # print(f"[OVERRIDE] EXECUTE({action.model_id}-{action.quant_type}) blocked: "
                    #       f"peak {total_peak:.0f}MB > limit {obs.ram_limit_mb}MB. Converting to EVICT({evict_m}-{evict_q}).", file=sys.stderr)
                    action.command = "EVICT"
                    action.model_id = evict_m
                    action.quant_type = evict_q
                    action.batch_size = 0
                else:
                    # print(f"[OVERRIDE] EXECUTE({action.model_id}-{action.quant_type}) blocked: "
                    #       f"peak {total_peak:.0f}MB > limit {obs.ram_limit_mb}MB. Emitting IDLE.", file=sys.stderr)
                    action.command = "IDLE"

    # EVICT override 
    elif action.command == "EVICT":
        if not obs.loaded_models:
            action.command = "IDLE"
            return action
        if action.model_id and action.quant_type:
            key = f"{action.model_id}-{action.quant_type}"
            if key not in obs.loaded_models:
                # Model not loaded — try to find it by model_id alone
                alt = loaded_key(obs, action.model_id)
                if alt:
                    parts = alt.rsplit("-", 1)
                    action.model_id   = parts[0]
                    action.quant_type = parts[1]
                else:
                    action.command = "IDLE"
        elif not action.model_id:
            # No model specified and multiple loaded → IDLE, let LLM decide
            if len(obs.loaded_models) > 1:
                action.command = "IDLE"
            else:
                only_key = list(obs.loaded_models.keys())[0]
                parts = only_key.rsplit("-", 1)
                action.model_id   = parts[0]
                action.quant_type = parts[1]

    return action


#  Roster string builder

def build_roster_str(obs: ModelFlowObservation) -> str:
    lines = ["  MODEL-QUANT          | TIER   | SIZE(MB) | GEN t/s | PROMPT t/s"]
    for role, info in obs.model_summary.items():
        for quant, stats in sorted(info["stats"].items()):
            lines.append(
                f"  {info['model_id']}-{quant:<10} | {stats['tier']:<6} | "
                f"{stats['size_mb']:>6}   | {stats['gen_tps']:>7.1f} | {stats['prompt_tps']:>9.1f}"
            )
    return "\n".join(lines)


# ── TERMINAL VISUALIZATION (RAM occupancy per step + action) ─────────────────────
def print_visualization(
    task_name: str,
    step_num: int,
    obs: ModelFlowObservation,
    action: Optional[ModelFlowAction] = None,
    reward: float = 0.0,
):
    """Beautiful terminal visualization of RAM occupancy, loaded models,
    action taken, queue, and progress. Shows how RAM changes after every step."""
    width = 90
    border = "=" * width
    print("\n" + border)
    
    if step_num == 0:
        print(f" TASK: {task_name.upper():^20} | INITIAL STATE ".center(width, "="))
    else:
        print(f" TASK: {task_name.upper():^20} | STEP {step_num:2d} ".center(width, "="))
    print(border)
    
    # Loaded models + total weight size
    if obs.loaded_models:
        total_mb = sum(v["size_mb"] for v in obs.loaded_models.values())
        print(f" LOADED MODELS : ({len(obs.loaded_models)} slots, {total_mb}MB total)")
        for key, slot in obs.loaded_models.items():
            gen_tps = slot.get("gen_tps", 0)
            print(f" → {key:<30} | tier={slot['tier']:<6} | {slot['size_mb']}MB | {gen_tps:.1f}t/s")
    else:
        print(f" LOADED MODELS : NONE")
    
    # RAM usage bar (replaces the original VRAM bar)
    ram_pct = min(100, int(obs.ram_used_mb / obs.ram_limit_mb * 100))
    bar = "█" * (ram_pct // 5) + "░" * (20 - ram_pct // 5)
    print(f" RAM USAGE     : {obs.ram_used_mb:5d} / {obs.ram_limit_mb} MB [{bar}] {ram_pct:3d}%")
    
    # Memory spike (if active)
    if obs.pressure_spike_mb > 0:
        print(
            f" MEMORY SPIKE  : {obs.pressure_spike_mb}MB active, "
            f"{obs.spike_steps_remaining} steps remaining"
        )
    
    # Action taken
    if step_num == 0:
        act_str = "ENVIRONMENT RESET (no action yet)"
    else:
        mod = action.model_id or ""
        quant = action.quant_type or ""
        act_str = (
            f"{action.command}({mod}-{quant}, batch={action.batch_size})"
            if mod else f"{action.command}(batch={action.batch_size})"
        )
    print(f" ACTION TAKEN  : {act_str}")
    print(f" REWARD        : {reward:+8.2f}")
    
    if obs.last_action_error:
        print(f" ERROR         : {obs.last_action_error}")
    elif obs.last_action_feedback:
        print(f" FEEDBACK      : {obs.last_action_feedback}")
    
    # Queue status
    print(f"\n QUEUE STATUS (pending: {len(obs.queue)})")
    for i, req in enumerate(obs.queue[:12]):
        typ = req.model_type.upper().ljust(10)
        cmplx = req.complexity.upper().ljust(10)
        print(f" {i+1:2d}. {req.request_id:>8} | {typ} | {cmplx} | age={req.age_steps:2d}")
    if len(obs.queue) > 12:
        print(f" ... +{len(obs.queue) - 12} more requests")
    
    # Progress
    completed = obs.info.get("completed", 0)
    print(f" PROGRESS      : {completed} completed | {len(obs.queue)} pending | Step: {step_num}")
    print(border + "\n")
def get_system_prompt(
    roster_str:    str,
    ram_limit_mb: int,
    q_stats:       Dict[str, Dict],
    obs:           ModelFlowObservation,
) -> str:
    # Pre-compute a concise RAM budget summary the LLM can reason over
    loaded_summary = []
    for key, slot in obs.loaded_models.items():
        loaded_summary.append(f"{key}: {slot['size_mb']}MB (tier={slot['tier']})")
    loaded_str = ", ".join(loaded_summary) if loaded_summary else "none"

    free_mb = ram_free(obs)

    # Pending work summary
    work_lines = []
    for mid, qs in q_stats.items():
        work_lines.append(
            f"  {mid}: {qs['total']} requests ({qs['reasoning']} reasoning, {qs['standard']} standard)"
        )
    work_str = "\n".join(work_lines) if work_lines else "  (queue empty)"

    return f"""You are an ML infrastructure orchestrator managing a GPU server.
Your goal: clear ALL queued requests with minimum steps and penalties.

HARDWARE
RAM limit:      {ram_limit_mb} MB
System overhead: {SYSTEM_OVERHEAD_MB} MB (always reserved)
Effective free:  {free_mb} MB right now
Currently loaded: {loaded_str}

ROLE → MODEL MAPPING 
  chatbot    → gemma-3-4b
  translator → llama_1b
  coder      → qwen3.5-2b

QUANT TIERS 
  Q4_K_M = low    → standard requests ONLY
  Q5_K_M = medium → standard requests ONLY  ← CANNOT serve reasoning
  Q6_K   = high   → standard + reasoning    ← minimum for reasoning
  Q8_0   = risky  → standard + reasoning

 PLANNING RULES (follow strictly) 

1. RAM CHECK FIRST: Before LOAD or REPLACE, verify:
     model_host_mb + current_ram_used + {SYSTEM_OVERHEAD_MB} (overhead) ≤ {ram_limit_mb}
   If it doesn't fit, EVICT something first.

2. EXECUTE PEAK RAM (CRITICAL): 
   EXECUTE creates a dynamic memory peak using the values from loaded_models:
   peak = ctx_mb + comp_mb + (kv_mb_max × batch_size / 8.0)
   Total RAM after EXECUTE = current_ram_used + peak + {SYSTEM_OVERHEAD_MB}
   If this would exceed ram_limit_mb, you MUST EVICT a non-needed model first.

3. QUANT SELECTION: If a model has ANY reasoning requests pending → load Q6_K.
   Never load Q5_K_M for a model with reasoning in the queue.

4. BATCH GREEDILY + MIXED REQUESTS (NON-NEGOTIABLE):
   Age penalty exists but is MODERATE. Large negative rewards on EXECUTE (e.g. -200 to -1600) are NORMAL for slow batches — IGNORE them and DO NOT EVICT.
   Always set batch_size = all pending requests for that model (max 8).
   If a model has BOTH reasoning AND standard requests, load Q6_K and KEEP IT LOADED until the entire queue for that model is completely empty.
   Only downgrade gemma-3-4b after ALL reasoning is cleared AND >40 standard requests remain.

5. AVOID PREMATURE EVICTION: Only evict a model when its queue section is FULLY drained.
   Do NOT evict just because you see a big negative reward on EXECUTE.

6. EXECUTE-AFTER-LOAD: After every LOAD or REPLACE, immediately EXECUTE that model.
   Do not load then idle.

7. REPLACE over EVICT+LOAD: When swapping quant on the same model, use REPLACE — it
   is faster and cheaper than EVICT followed by LOAD.

PENDING WORK
{work_str}

AVAILABLE CONFIGS
{roster_str}

ACTIONS
  LOAD(model_id, quant_type)
  EXECUTE(model_id, quant_type, batch_size)
  EVICT(model_id, quant_type)
  REPLACE(model_id, quant_type, evict_model_id, evict_quant_type)
  IDLE  — costs −15, avoid unless blocked

Respond ONLY with valid JSON (no markdown):
{{"command": "LOAD"|"EXECUTE"|"EVICT"|"REPLACE"|"IDLE",
  "model_id": str|null, "quant_type": str|null, "batch_size": int,
  "evict_model_id": str|null, "evict_quant_type": str|null}}"""

# Observation → text 

def observation_to_text(obs: ModelFlowObservation, q_stats: Dict[str, Dict]) -> str:
    # Outcome
    if obs.last_action_error:
        outcome = f"LAST ACTION FAILED: {obs.last_action_error}"
    elif obs.last_action_feedback:
        outcome = f"LAST ACTION OK: {obs.last_action_feedback}"
    else:
        outcome = "READY."

    # RAM state
    free_mb   = ram_free(obs)
    spike_str = (f" | Spike +{obs.pressure_spike_mb}MB ({obs.spike_steps_remaining} steps)"
                 if obs.pressure_spike_mb > 0 else "")
    ram_line = (f"RAM: {obs.ram_used_mb}/{obs.ram_limit_mb}MB used "
                 f"(free={free_mb}MB){spike_str}")

    # Loaded models with effective gen_tps
    sum_cpu    = sum(m.get("cpu_avg", 0) for m in obs.loaded_models.values())
    contention = min(0.8, sum_cpu / 400.0)
    loaded     = []
    for key, v in obs.loaded_models.items():
        eff_tps = v.get("gen_tps", 0) * (1.0 - contention)
        loaded.append(f"{key}(tier={v['tier']}, {v['size_mb']}MB, {eff_tps:.1f}t/s)")
    loaded_str = "LOADED: " + (", ".join(loaded) if loaded else "NONE")

    # Pending work per model (including reasoning flag)
    work_parts = []
    for mid, qs in q_stats.items():
        work_parts.append(
            f"{mid}: {qs['total']} pending ({qs['reasoning']} reasoning)"
        )
    work_str = "QUEUE: " + (" | ".join(work_parts) if work_parts else "EMPTY")

    # Top queue items for context
    top_items  = ", ".join(
        f"{r.request_id}:{r.model_type[:4]}/{r.complexity[:3]}/age={r.age_steps}"
        for r in obs.queue[:8]
    )
    top_str    = f"TOP8: [{top_items}]" if top_items else ""

    # Grader metrics
    metrics = obs.info.get("grader_metrics", {})
    meta    = (f"Step={obs.step_count} completed={obs.info.get('completed',0)} "
               f"loads={metrics.get('loads',0)} evicts={metrics.get('evicts',0)} "
               f"ooms={metrics.get('ooms',0)} idles={metrics.get('idles',0)}")

    return "\n".join(filter(None, [outcome, ram_line, loaded_str, work_str, top_str, meta]))


# Context helpers 

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def compress_step(
    step_num: int,
    action:   ModelFlowAction,
    reward:   float,
    feedback: Optional[str],
    error:    Optional[str],
) -> str:
    mod    = action.model_id   or ""
    quant  = action.quant_type or ""
    result = f"ERR:{error[:80]}" if error else (feedback[:80] if feedback else "OK")
    return f"S{step_num}: {action.command}({mod}-{quant}) b={action.batch_size} R{reward:+.1f} | {result}"


def build_messages(
    system_prompt:      str,
    compressed_history: List[str],
    current_obs_text:   str,
) -> List[Dict]:
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]
    if compressed_history:
        history_block = "STEP HISTORY (recent):\n" + "\n".join(
            f"  {line}" for line in compressed_history
        )
        messages.append({"role": "user",      "content": history_block})
        messages.append({"role": "assistant", "content": "Acknowledged."})
    messages.append({"role": "user", "content": current_obs_text})
    return messages


# Task runner

def run_task(task_name: str):
    env  = ModelFlowEnvironment()
    obs  = env.reset(task_name=task_name)

    step_num            = 0
    rewards             = []
    done                = False
    compressed_history: List[str] = []
    score               = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={active_model}", flush=True)

    # INITIAL VISUALIZATION (step 0)
    print_visualization(task_name, 0, obs)

    try:
        while not done and step_num < MAX_STEPS_PER_TASK:
            step_num += 1

            # Compute derived context 
            q_stats_now   = queue_stats(obs)
            roster_str    = build_roster_str(obs)
            system_prompt = get_system_prompt(roster_str, obs.ram_limit_mb, q_stats_now, obs)
            obs_text      = observation_to_text(obs, q_stats_now)
            recent        = compressed_history[-CONTEXT_HISTORY_STEPS:]
            messages      = build_messages(system_prompt, recent, obs_text)

        
            action_dict = {"command": "IDLE", "model_id": None, "quant_type": None,
                           "batch_size": 1, "evict_model_id": None, "evict_quant_type": None}
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=active_model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        response_format={"type": "json_object"} if USE_GROQ_ONLY else None,
                    )
                    raw = response.choices[0].message.content.strip()
                    print(f"[LLM RAW]: {raw}", file=sys.stderr)
                    action_dict = parse_action(raw)
                    break
                except Exception as e:
                    err_str = str(e)
                    wait    = BASE_BACKOFF_S * (2 ** attempt)
                    if any(kw in err_str.lower() for kw in ("rate limit", "429", "too many")) \
                            and attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                    else:
                        break

            # Build ModelFlowAction 
            action = ModelFlowAction(
                command         = action_dict.get("command", "IDLE"),
                model_id        = action_dict.get("model_id"),
                quant_type      = action_dict.get("quant_type"),
                batch_size      = min(action_dict.get("batch_size", 8), 8),
                evict_model_id  = action_dict.get("evict_model_id"),
                evict_quant_type= action_dict.get("evict_quant_type"),
            )


            action = apply_planning_override(action, obs)

            # Step environment
            obs        = env.step(action)
            reward_val = round(obs.reward, 2)
            rewards.append(reward_val)
            done = obs.done

            # DETAILED RAM VISUALIZATION after every action
            print_visualization(task_name, step_num, obs, action, reward_val)

            done_str  = "true" if done else "false"
            error_val = obs.last_action_error if obs.last_action_error else "null"
            act_str   = f"{action.command}[{action.model_id or ''}"
            model_quant = f"{act_str}-{action.quant_type}]"
            print(
                f"[STEP] step={step_num} action={model_quant} reward={reward_val:.2f} "
                f"done={done_str} error={error_val}",
                flush=True,
            )

            compressed_history.append(
                compress_step(step_num, action, obs.reward,
                              obs.last_action_feedback, obs.last_action_error)
            )
        
        score = env.score_task()
    
    finally:
        success     = len(obs.queue) == 0
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    for task in TASKS:
        try:
            run_task(task)
            # print("\n" + "=" * 80 + "\n", file=sys.stderr)
        except Exception as e:
            # print(f"Error running task {task}: {e}", file=sys.stderr)
            pass