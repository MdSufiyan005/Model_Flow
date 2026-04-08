from typing import Dict

from helpers.tools import get_model_size, can_load as tool_can_load
from models import ModelFlowObservation


def ram_free(obs: ModelFlowObservation) -> int:
    return int(
        obs.info.get(
            "ram_free_mb",
            obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb,
        )
    )


def model_host_mb(obs: ModelFlowObservation, model_id: str, quant_type: str) -> int:
    result = get_model_size(model_id, quant_type, obs)
    if "error" in result:
        return 99999
    return int(result["host_mb"])


def can_load(obs: ModelFlowObservation, model_id: str, quant_type: str) -> bool:
    result = tool_can_load(model_id, quant_type, obs)
    return bool(result.get("fits", False))