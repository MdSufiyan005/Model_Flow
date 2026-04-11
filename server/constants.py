from typing import Dict

ACTIVE_MODELS = {"qwen3.5-2b", "llama_1b", "gemma-3-4b"}
QUANTS = {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}

QUANT_TO_TIER: Dict[str, str] = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K":   "high",
    "Q8_0":   "risky",
}

TIER_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "risky": 2}
COMPLEXITY_MIN_RANK: Dict[str, int] = {"standard": 0, "reasoning": 2}

ROLE_TO_MODEL: Dict[str, str] = {
    "chatbot":    "gemma-3-4b",
    "translator": "llama_1b",
    "coder":      "qwen3.5-2b",
}

# ── V2: SLA / non-stationarity constants ──────────────────────────────────────

# Base SLA window (steps). A request served with age_steps > this is "late".
SLA_BASE_STEPS: int = 40

# How many steps before SLA tightens by SLA_TIGHTEN_BY on hard/extreme tasks.
SLA_TIGHTEN_EVERY: int = 4

# How much the SLA shrinks each tightening cycle.
SLA_TIGHTEN_BY: int = 2

# Minimum SLA floor — never tightens below this.
SLA_FLOOR_STEPS: int = 10

# After T_shift steps (sampled per episode), the demand distribution flips.
# Agent observes "shift_detected" hint only after DEMAND_HINT_DELAY extra steps.
DEMAND_HINT_DELAY: int = 2

# Heat thresholds for bucketed display in observation.
# raw heat → bucket: 0-1 = "low", 2-3 = "medium", 4+ = "high"
HEAT_BUCKET_LOW    = 1   # inclusive upper bound for "low"
HEAT_BUCKET_MEDIUM = 3   # inclusive upper bound for "medium"
# above HEAT_BUCKET_MEDIUM → "high"

# ── V2: Quality normalisers (from real benchmark data) ────────────────────────
# Max BLEU and ROUGE-L observed across all 12 profiled model/quant combos.
# Used in rewards.py to normalise quality scores to [0, 1].
MAX_BLEU_OBSERVED:   float = 36.60   # gemma-3-4b Q4_K_M
MAX_ROUGE_OBSERVED:  float = 35.14   # llama_1b Q8_0

# Quality penalty applied to reasoning requests at each quant tier.
# Derived from the perplexity delta observed in benchmark data.
# gemma perplexity rises from 1.069 (Q8_0) to 1.081 (Q4_K_M) — small delta.
# llama perplexity rises from 1.034 (Q8_0) to 1.033 (Q4_K_M) — near flat.
# We apply a steeper reasoning penalty because reasoning is more sensitive to
# quantisation noise than perplexity alone captures.
REASONING_QUANT_PENALTY: Dict[str, float] = {
    "Q4_K_M": 0.35,
    "Q5_K_M": 0.18,
    "Q6_K":   0.06,
    "Q8_0":   0.00,
}

# ── V2: Model heat decay ──────────────────────────────────────────────────────
# Probability of a quality failure = min(HEAT_FAIL_SLOPE * heat, HEAT_FAIL_CAP)
# heat = number of times this (model, quant) key has been loaded since episode start.
HEAT_FAIL_SLOPE: float = 0.07   # +7% failure probability per load
HEAT_FAIL_CAP:   float = 0.55   # never exceeds 55% failure probability

# ── Tasks ─────────────────────────────────────────────────────────────────────
TASKS = {
    "single-load": {
        "requests": [{"model_type": "chatbot", "complexity": "standard"}] * 9,
        "sla_tightening": False,
        "demand_shift":   False,
    },
    "multi-load": {
        "requests": [
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "coder",      "complexity": "reasoning"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "reasoning"},
        ],
        "sla_tightening": False,
        "demand_shift":   True,    # T_shift sampled ∈ [5, 8]
    },
    "quality-limit": {
        "requests": [
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "chatbot",    "complexity": "reasoning"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "reasoning"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "coder",      "complexity": "reasoning"},
        ],
        "sla_tightening": True,    # SLA tightens every SLA_TIGHTEN_EVERY steps
        "demand_shift":   False,
    },
    "ram-pressure": {
        "requests": [
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "coder",      "complexity": "reasoning"},
            {"model_type": "chatbot",    "complexity": "reasoning"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "coder",      "complexity": "reasoning"},
            {"model_type": "chatbot",    "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "coder",      "complexity": "standard"},
            {"model_type": "chatbot",    "complexity": "reasoning"},
            {"model_type": "translator", "complexity": "standard"},
        ],
        "sla_tightening": True,
        "demand_shift":   True,    # compound: shift + tightening + spikes
    },
}