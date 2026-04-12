"""
Loads the benchmark JSON produced by the profiling repo and normalises it
into a flat roster dict keyed by "{model_id}-{quant}".

"""

from __future__ import annotations
import json
from typing import Dict


# Mapping from model_name substrings to canonical model_id
_MODEL_NAME_MAP = {
    "Qwen3.5-2B": "qwen3.5-2b",
    "Qwen_Qwen3.5": "qwen3.5-2b",
    "Llama-3.2-1B": "llama_1b",
    "gemma-3-4b": "gemma-3-4b",
    "google_gemma-3-4b": "gemma-3-4b",
}

_QUANT_KEY_MAP = {
    "Q4_K_M": "Q4_K_M",
    "Q5_K_M": "Q5_K_M",
    "Q6_K":   "Q6_K",
    "Q8_0":   "Q8_0",
}

_TIER_MAP = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K":   "high",
    "Q8_0":   "risky",
}


def _parse_range(range_str: str) -> tuple:
    """Parse "lo-hi" string into (lo, hi) float tuple."""
    try:
        parts = range_str.split("-")
        return float(parts[0]), float(parts[1])
    except Exception:
        return (0.0, 0.0)


def _infer_model_id(model_key: str, model_name: str) -> str | None:
    combined = model_key + " " + model_name
    for substr, model_id in _MODEL_NAME_MAP.items():
        if substr.lower() in combined.lower():
            return model_id
    return None


def _infer_quant(model_key: str, model_name: str) -> str | None:
    combined = model_key + " " + model_name
    for quant in _QUANT_KEY_MAP:
        if quant in combined:
            return quant
    return None


def load_roster(json_path: str) -> Dict[str, dict]:
    """
    Returns a dict keyed by "{model_id}-{quant}" with normalised fields
    ready for consumption by the environment and reward functions.
    """
    with open(json_path) as f:
        raw = json.load(f)

    roster: Dict[str, dict] = {}

    for entry in raw.get("models", []):
        model_key  = entry.get("model_key", "")
        model_name = entry.get("model_name", "")

        model_id = _infer_model_id(model_key, model_name)
        quant    = _infer_quant(model_key, model_name)

        if model_id is None or quant is None:
            continue

        key = f"{model_id}-{quant}"

        mem   = entry.get("memory", {})
        speed = entry.get("speed", {})
        qual  = entry.get("quality", {})
        cpu   = entry.get("cpu", {})

        # ── V2: parse range strings into (lo, hi) tuples ──────────────────
        load_range_ms = _parse_range(speed.get("load_time_range_ms", "0-0"))
        host_mb_range = _parse_range(mem.get("host_total_mb_range", "0-0"))

        roster[key] = {
            # Identity
            "model":  model_id,
            "quant":  quant,
            "tier":   _TIER_MAP.get(quant, "low"),

            # RAM — average and range for stochastic sampling
            "host_mb":       mem.get("host_total_mb_avg", entry.get("weight_size_mb", 0)),
            "host_mb_range": host_mb_range,
            "size_mb":       entry.get("weight_size_mb", 0),
            "ctx_mb":        mem.get("context_mb_avg", 0),
            "comp_mb":       mem.get("compute_mb_avg", 0),
            "kv_mb_max":     mem.get("kv_total_mb_max", 0),
            "peak_rss_mb":   mem.get("peak_rss_mb_max", 0),

            # Load time — average, variance, and range for stochastic sampling
            "load_avg_ms":  speed.get("load_time_avg_ms", 500),
            "load_var_ms":  speed.get("load_time_variance_ms", 0),
            "load_range_ms": load_range_ms,

            # Throughput
            "gen_tps":    speed.get("gen_tps", 10.0),
            "prompt_tps": speed.get("prompt_tps", 30.0),

            # CPU
            "cpu_avg": cpu.get("avg_cpu_percent", 300.0),
            "cpu_max": cpu.get("max_cpu_percent", 400.0),

            # Quality — used in V2 reward function
            "bleu_avg":   qual.get("bleu_avg", 0.0),
            "rouge_l_avg": qual.get("rouge_l_avg", 0.0),
            "perplexity":  qual.get("perplexity", 1.05),
        }

    return roster