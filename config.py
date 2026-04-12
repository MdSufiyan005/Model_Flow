"""
config.py — shared constants.

"""

import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# ── Re-read the same env vars so helpers can import them without circular deps ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct:together")
HF_TOKEN     = os.getenv("HF_TOKEN")

active_model = MODEL_NAME

BENCHMARK = "modelflow"
TASKS     = ["single-load", "multi-load", "quality-limit", "ram-pressure"]

# Keep well under the grader's optimal_cutoff=8 to maximise efficiency score.
MAX_STEPS_PER_TASK = 15

TEMPERATURE           = 0.05   # near-deterministic for consistent decisions
MAX_TOKENS            = 150 #800 -rate limit
CONTEXT_HISTORY_STEPS = 6
MAX_RETRIES           = 4
BASE_BACKOFF_S        = 2.0
SYSTEM_OVERHEAD_MB    = 1100
EXEC_SAFETY_BUFFER_MB = 2000
# Never load if free RAM after load would drop below this value.
# During ram-pressure tasks spikes can reach 2 500 MB — leave room.
RAM_SAFETY_BUFFER_MB = 1600

# model mapping
ROLE_TO_MODEL: Dict[str, str] = {
    "chatbot":    "gemma-3-4b",
    "translator": "llama_1b",
    "coder":      "qwen3.5-2b",
}

# Quantisation constants
QUANT_TIER: Dict[str, str] = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K":   "high",
    "Q8_0":   "risky",
}

TIER_RANK: Dict[str, int] = {
    "low":    1,
    "medium": 2,
    "high":   3,
    "risky":  4,
}

COMPLEXITY_MIN_RANK: Dict[str, int] = {
    "simple":    1,
    "standard":  1,   # alias used in some request payloads
    "reasoning": 3,
}

REASONING_QUANTS    = {"Q6_K", "Q8_0"}
REASONING_MIN_QUANT = "Q6_K"          # cheapest quant that satisfies reasoning
DEFAULT_SIMPLE_QUANT    = "Q4_K_M"
DEFAULT_REASONING_QUANT = "Q6_K"

QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]