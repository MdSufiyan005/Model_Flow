"""
tools.py - Agent introspection tools for ModelFlow inference.

Three tools are exposed to the LLM via OpenAI function-calling:
  • get_model_size       → host RAM (MB) for a model+quant pair
  • can_load             → RAM pre-flight check before LOAD / REPLACE
  • simulate_execute_peak → peak RAM prediction before EXECUTE

The current observation is injected at call time by `dispatch_tool_call`;
the LLM only passes model_id / quant_type / batch_size in its arguments.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from models import ModelFlowObservation  # adjust import path if needed

SYSTEM_OVERHEAD_MB = 1100  # must match environment constant


# ── Pure logic ────────────────────────────────────────────────────────────────

def get_model_size(
    model_id: str,
    quant_type: str,
    obs: ModelFlowObservation,
) -> Dict[str, Any]:
    """
    Return host RAM (MB) needed to load `model_id`-`quant_type`.

    Returns
    -------
    On success:
        {"model_id": str, "quant_type": str, "host_mb": int}
    On failure:
        {"error": str}
    """
    for _role, info in obs.model_summary.items():
        if info["model_id"] == model_id:
            stats = info["stats"].get(quant_type)
            if stats:
                return {
                    "model_id":   model_id,
                    "quant_type": quant_type,
                    "host_mb":    stats["size_mb"],
                    "tier":       stats["tier"],
                    "gen_tps":    stats["gen_tps"],
                }
    return {"error": f"Unknown config: {model_id}-{quant_type}"}


def can_load(
    model_id: str,
    quant_type: str,
    obs: ModelFlowObservation,
) -> Dict[str, Any]:
    """
    Pre-flight RAM check: will `model_id`-`quant_type` fit right now?

    Returns
    -------
    {
      "model_id":    str,
      "quant_type":  str,
      "host_mb":     int,   # size of the model
      "ram_used_mb": int,   # current RAM in use
      "spike_mb":    int,   # active memory spike
      "ram_free_mb": int,   # effective free RAM (after spike + overhead)
      "fits":        bool,  # True → safe to LOAD
      "deficit_mb":  int,   # 0 if fits, otherwise how many MB short
    }
    On unknown model:
        {"error": str}
    """
    size_result = get_model_size(model_id, quant_type, obs)
    if "error" in size_result:
        return size_result

    host_mb  = size_result["host_mb"]
    free_mb  = int(obs.info.get(
        "ram_free_mb",
        obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb - SYSTEM_OVERHEAD_MB,
    ))
    fits     = host_mb <= free_mb
    return {
        "model_id":    model_id,
        "quant_type":  quant_type,
        "host_mb":     host_mb,
        "ram_used_mb": int(obs.ram_used_mb),
        "spike_mb":    int(obs.pressure_spike_mb),
        "ram_free_mb": free_mb,
        "fits":        fits,
        "deficit_mb":  max(0, host_mb - free_mb),
    }


def simulate_execute_peak(
    model_id: str,
    quant_type: str,
    batch_size: int,
    obs: ModelFlowObservation,
) -> Dict[str, Any]:
    """
    Predict peak RAM during EXECUTE before issuing the command.

    Peak formula (mirrors the environment):
        dynamic_peak = ctx_mb + comp_mb + (kv_mb_max × batch_size / 8)
        total_peak   = ram_used_mb + spike_mb + SYSTEM_OVERHEAD + dynamic_peak

    Returns
    -------
    {
      "model_id":         str,
      "quant_type":       str,
      "batch_size":       int,
      "dynamic_peak_mb":  int,  # extra RAM the execution will consume
      "total_ram_peak_mb":int,  # total RAM at execution peak
      "ram_limit_mb":     int,
      "headroom_mb":      int,  # positive = safe, negative = would OOM
      "safe":             bool,
    }
    On unknown / unloaded model:
        {"error": str}
    """
    key  = f"{model_id}-{quant_type}"
    slot = obs.loaded_models.get(key)
    if not slot:
        return {"error": f"{key} is not currently loaded"}

    ctx     = slot.get("ctx_mb",    0.0)
    comp    = slot.get("comp_mb",   0.0)
    kv_max  = slot.get("kv_mb_max", 0.0)
    kv_dyn  = kv_max * (max(1, min(batch_size, 8)) / 8.0)

    dynamic_peak  = ctx + comp + kv_dyn
    total_peak    = obs.ram_used_mb + obs.pressure_spike_mb + SYSTEM_OVERHEAD_MB + dynamic_peak
    headroom      = obs.ram_limit_mb - total_peak

    return {
        "model_id":          model_id,
        "quant_type":        quant_type,
        "batch_size":        batch_size,
        "dynamic_peak_mb":   round(dynamic_peak),
        "total_ram_peak_mb": round(total_peak),
        "ram_limit_mb":      obs.ram_limit_mb,
        "headroom_mb":       round(headroom),
        "safe":              headroom >= 0,
    }


# ── Dispatcher (called by the inference loop) ─────────────────────────────────

def dispatch_tool_call(
    tool_name: str,
    arguments: str | Dict,
    obs: ModelFlowObservation,
) -> str:
    """
    Execute a tool call from the LLM and return a JSON string result.

    Parameters
    ----------
    tool_name : str         - one of get_model_size / can_load / simulate_execute_peak
    arguments : str | dict  - raw JSON string or already-parsed dict from the LLM
    obs       : ModelFlowObservation - current environment observation (injected)

    Returns
    -------
    JSON string to be sent back as a tool result message.
    """
    args: Dict = json.loads(arguments) if isinstance(arguments, str) else arguments

    if tool_name == "get_model_size":
        result = get_model_size(
            model_id   = args["model_id"],
            quant_type = args["quant_type"],
            obs        = obs,
        )

    elif tool_name == "can_load":
        result = can_load(
            model_id   = args["model_id"],
            quant_type = args["quant_type"],
            obs        = obs,
        )

    elif tool_name == "simulate_execute_peak":
        result = simulate_execute_peak(
            model_id   = args["model_id"],
            quant_type = args["quant_type"],
            batch_size = int(args.get("batch_size", 4)),
            obs        = obs,
        )

    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)


# ── OpenAI / Groq tool schemas ────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_size",
            "description": (
                "Returns the host RAM (MB) required to load a specific model+quant combination, "
                "along with its tier and generation speed. Use this before LOAD or REPLACE to "
                "know exactly how large a model is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier, e.g. 'gemma-3-4b', 'llama_1b', 'qwen3.5-2b'.",
                    },
                    "quant_type": {
                        "type": "string",
                        "enum": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                        "description": "Quantization level.",
                    },
                },
                "required": ["model_id", "quant_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "can_load",
            "description": (
                "Pre-flight RAM check: tells you whether loading model_id-quant_type will fit "
                "in current free RAM (accounting for system overhead and any active memory spike). "
                "Always call this before issuing LOAD or REPLACE to avoid OOM penalties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier.",
                    },
                    "quant_type": {
                        "type": "string",
                        "enum": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                        "description": "Quantization level to test.",
                    },
                },
                "required": ["model_id", "quant_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_execute_peak",
            "description": (
                "Simulates peak RAM usage during an EXECUTE command. "
                "Returns total_ram_peak_mb and whether it is safe (headroom_mb >= 0). "
                "Call this before issuing EXECUTE to confirm you won't hit a runtime OOM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier of the loaded model to execute.",
                    },
                    "quant_type": {
                        "type": "string",
                        "enum": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                        "description": "Quant type of the loaded model.",
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Number of requests in the batch (1–8).",
                    },
                },
                "required": ["model_id", "quant_type", "batch_size"],
            },
        },
    },
]