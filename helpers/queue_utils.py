import json
from typing import Dict, Optional, Tuple

from config import ROLE_TO_MODEL, REASONING_MIN_QUANT, REASONING_QUANTS
from models import ModelFlowObservation

_KNOWN_MODELS = {"gemma-3-4b", "llama_1b", "qwen3.5-2b"}
_KNOWN_QUANTS = {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}


def queue_stats(obs: ModelFlowObservation) -> Dict[str, Dict]:
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
    qs = queue_stats(obs)
    info = qs.get(model_id, {})
    return REASONING_MIN_QUANT if info.get("reasoning", 0) > 0 else "Q4_K_M"


def loaded_key(obs: ModelFlowObservation, model_id: str) -> Optional[str]:
    for key in obs.loaded_models:
        if obs.loaded_models[key]["model"] == model_id:
            return key
    return None


def loaded_quant(obs: ModelFlowObservation, model_id: str) -> Optional[str]:
    key = loaded_key(obs, model_id)
    return obs.loaded_models[key]["quant"] if key else None


def can_serve_reasoning(obs: ModelFlowObservation, model_id: str) -> bool:
    q = loaded_quant(obs, model_id)
    return q in REASONING_QUANTS if q else False


def _normalise_model_id(raw_model, raw_quant) -> Tuple[Optional[str], Optional[str]]:
    if not raw_model:
        return raw_model, raw_quant

    model = raw_model.strip().strip('"').strip("'")

    if not raw_quant:
        for q in _KNOWN_QUANTS:
            if model.endswith(f"-{q}"):
                return model[: -(len(q) + 1)], q

    if model not in _KNOWN_MODELS:
        for m in _KNOWN_MODELS:
            if m.startswith(model) or model.startswith(m.split("-")[0]):
                model = m
                break

    quant = raw_quant.strip().strip('"').strip("'") if raw_quant else raw_quant
    if quant and quant not in _KNOWN_QUANTS:
        quant = None

    return model, quant


def parse_action(response_text: str) -> Dict:
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")

        data = json.loads(response_text[start:end])
        command = str(data.get("command", "IDLE")).upper().strip()
        if command not in {"LOAD", "EXECUTE", "EVICT", "IDLE", "REPLACE"}:
            command = "IDLE"

        raw_model, raw_quant = _normalise_model_id(
            data.get("model_id"),
            data.get("quant_type"),
        )
        evict_model, evict_quant = _normalise_model_id(
            data.get("evict_model_id"),
            data.get("evict_quant_type"),
        )

        batch_size = data.get("batch_size")
        if batch_size is None or not isinstance(batch_size, (int, float)):
            batch_size = 4
        batch_size = max(1, min(int(batch_size), 8))

        return {
            "command": command,
            "model_id": raw_model,
            "quant_type": raw_quant,
            "batch_size": batch_size,
            "evict_model_id": evict_model,
            "evict_quant_type": evict_quant,
        }
    except Exception:
        return {
            "command": "IDLE",
            "model_id": None,
            "quant_type": None,
            "batch_size": 1,
            "evict_model_id": None,
            "evict_quant_type": None,
        }


def get_eviction_target(obs, exclude_model_id=None):
    q_stats = queue_stats(obs)
    candidates = []

    for key, slot in obs.loaded_models.items():
        mid = slot["model"]
        if mid == exclude_model_id:
            continue
        pending = q_stats.get(mid, {}).get("total", 0)
        candidates.append(
            {
                "model_id": mid,
                "quant_type": slot["quant"],
                "size": slot.get("size_mb", 0),
                "needed": pending > 0,
            }
        )

    if not candidates:
        return None, None

    candidates.sort(key=lambda c: (c["needed"], -c["size"]))
    best = candidates[0]
    return best["model_id"], best["quant_type"]