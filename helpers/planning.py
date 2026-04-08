from config import REASONING_QUANTS, SYSTEM_OVERHEAD_MB
from helpers.model_utils import can_load, model_host_mb, ram_free
from helpers.queue_utils import (
    can_serve_reasoning,
    get_eviction_target,
    loaded_key,
    loaded_quant,
    queue_stats,
    required_quant,
)
from models import ModelFlowAction, ModelFlowObservation


def apply_planning_override(action: ModelFlowAction, obs: ModelFlowObservation) -> ModelFlowAction:
    q_stats = queue_stats(obs)

    if action.command == "LOAD":
        if not action.model_id or not action.quant_type:
            action.command = "IDLE"
            return action

        if loaded_key(obs, action.model_id):
            current_quant = loaded_quant(obs, action.model_id)
            needed_quant = required_quant(action.model_id, obs)
            if current_quant != needed_quant:
                action.command = "REPLACE"
                action.evict_model_id = action.model_id
                action.evict_quant_type = current_quant
                action.quant_type = needed_quant
            else:
                action.command = "IDLE"
            return action

        needed_quant = required_quant(action.model_id, obs)
        if action.quant_type not in REASONING_QUANTS and needed_quant in REASONING_QUANTS:
            action.quant_type = needed_quant

        if not can_load(obs, action.model_id, action.quant_type):
            model_qs = q_stats.get(action.model_id, {})
            has_reason = model_qs.get("reasoning", 0) > 0
            fallback = {"Q8_0": "Q6_K", "Q6_K": "Q5_K_M", "Q5_K_M": "Q4_K_M"}
            fb_quant = fallback.get(action.quant_type)

            if fb_quant and not has_reason and can_load(obs, action.model_id, fb_quant):
                action.quant_type = fb_quant
            else:
                evict_m, evict_q = get_eviction_target(obs, exclude_model_id=action.model_id)
                if evict_m:
                    action.command = "EVICT"
                    action.model_id = evict_m
                    action.quant_type = evict_q
                    action.batch_size = 0
                else:
                    action.command = "IDLE"

    elif action.command == "REPLACE":
        if not action.model_id or not action.quant_type:
            action.command = "IDLE"
            return action

        needed_quant = required_quant(action.model_id, obs)
        if action.quant_type not in REASONING_QUANTS and needed_quant in REASONING_QUANTS:
            action.quant_type = needed_quant

        if not action.evict_model_id and not action.evict_quant_type:
            action.evict_model_id = action.model_id
            action.evict_quant_type = loaded_quant(obs, action.model_id)

        evict_size = 0
        if action.evict_model_id and action.evict_quant_type:
            evk = f"{action.evict_model_id}-{action.evict_quant_type}"
            if evk in obs.loaded_models:
                evict_size = obs.loaded_models[evk].get("size_mb", 0)

        simulated_free = ram_free(obs) + evict_size
        needed_mb = model_host_mb(obs, action.model_id, action.quant_type)

        if needed_mb > simulated_free:
            evict_m, evict_q = get_eviction_target(obs, exclude_model_id=action.model_id)
            if evict_m:
                action.command = "EVICT"
                action.model_id = evict_m
                action.quant_type = evict_q
                action.evict_model_id = None
                action.evict_quant_type = None
                action.batch_size = 0
            else:
                action.command = "IDLE"

    elif action.command == "EXECUTE":
        if not action.model_id or not action.quant_type:
            if len(obs.loaded_models) == 1:
                only = list(obs.loaded_models.values())[0]
                action.model_id = only["model"]
                action.quant_type = only["quant"]
            else:
                action.command = "IDLE"
                return action

        model_qs = q_stats.get(action.model_id, {})
        if model_qs.get("total", 0) == 0:
            action.command = "IDLE"
            return action

        if model_qs.get("reasoning", 0) > 0 and not can_serve_reasoning(obs, action.model_id):
            action.command = "IDLE"
            return action

        pending_total = model_qs.get("total", 8)
        action.batch_size = min(action.batch_size, pending_total, 8)

        exec_key = f"{action.model_id}-{action.quant_type}"
        slot = obs.loaded_models.get(exec_key)
        if slot:
            kv_dyn = slot.get("kv_mb_max", 0) * (action.batch_size / 8.0)
            peak_dyn = slot.get("ctx_mb", 0) + slot.get("comp_mb", 0) + kv_dyn
            total_peak = obs.ram_used_mb + obs.pressure_spike_mb + SYSTEM_OVERHEAD_MB + peak_dyn

            if total_peak > obs.ram_limit_mb:
                evict_m, evict_q = get_eviction_target(obs, exclude_model_id=action.model_id)
                if evict_m:
                    action.command = "EVICT"
                    action.model_id = evict_m
                    action.quant_type = evict_q
                    action.batch_size = 0
                else:
                    action.command = "IDLE"

    elif action.command == "EVICT":
        if not obs.loaded_models:
            action.command = "IDLE"
            return action

        if action.model_id and action.quant_type:
            key = f"{action.model_id}-{action.quant_type}"
            if key not in obs.loaded_models:
                alt = loaded_key(obs, action.model_id)
                if alt:
                    parts = alt.rsplit("-", 1)
                    action.model_id = parts[0]
                    action.quant_type = parts[1]
                else:
                    action.command = "IDLE"
        elif not action.model_id:
            if len(obs.loaded_models) > 1:
                action.command = "IDLE"
            else:
                only_key = list(obs.loaded_models.keys())[0]
                parts = only_key.rsplit("-", 1)
                action.model_id = parts[0]
                action.quant_type = parts[1]

    return action