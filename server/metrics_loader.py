import json
import os
from typing import Dict

from server.constants import ACTIVE_MODELS, QUANTS, QUANT_TO_TIER


def load_roster(benchmark_json: str) -> Dict[str, dict]:
    """
    Load model metrics into the roster dictionary.

    The function first tries the provided path, then a fallback relative path
    so it works both locally and inside the container layout.
    """
    if not os.path.exists(benchmark_json):
        alt_path = benchmark_json.replace("model_flow/", "")
        if os.path.exists(alt_path):
            benchmark_json = alt_path

    roster: Dict[str, dict] = {}

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "Data",
        "combined_model_metrics.json",
    )

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        for m in data["models"]:
            parts = m["model_key"].rsplit("-", 1)
            model_name = parts[0]
            quant_type = parts[1]

            if model_name in ACTIVE_MODELS and quant_type in QUANTS:
                roster[m["model_key"]] = {
                    "model": model_name,
                    "quant": quant_type,
                    "tier": QUANT_TO_TIER[quant_type],
                    "size_mb": m["weight_size_mb"],
                    "host_mb": m["memory"]["host_total_mb_avg"],
                    "ctx_mb": m["memory"]["context_mb_avg"],
                    "comp_mb": m["memory"]["compute_mb_avg"],
                    "kv_mb_max": m["memory"]["kv_total_mb_max"],
                    "prompt_tps": m["speed"]["prompt_tps"],
                    "gen_tps": m["speed"]["gen_tps"],
                    "load_avg_ms": m["speed"]["load_time_avg_ms"],
                    "cpu_avg": m["cpu"]["avg_cpu_percent"],
                }
    else:
        roster = {
            "gemma-3-4b-Q4_K_M": {
                "model": "gemma-3-4b",
                "quant": "Q4_K_M",
                "tier": "low",
                "host_mb": 2583.0,
                "size_mb": 2300,
                "ctx_mb": 40.0,
                "comp_mb": 110.0,
                "kv_mb_max": 96.0,
                "prompt_tps": 120.0,
                "gen_tps": 24.0,
                "load_avg_ms": 1000.0,
                "cpu_avg": 30.0,
            },
        }

    return roster