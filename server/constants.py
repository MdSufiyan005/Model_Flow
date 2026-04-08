from typing import Dict

ACTIVE_MODELS = {"qwen3.5-2b", "llama_1b", "gemma-3-4b"}
QUANTS = {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}

QUANT_TO_TIER: Dict[str, str] = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K": "high",
    "Q8_0": "risky",
}

TIER_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "risky": 2}
COMPLEXITY_MIN_RANK: Dict[str, int] = {"standard": 0, "reasoning": 2}

ROLE_TO_MODEL: Dict[str, str] = {
    "chatbot": "gemma-3-4b",
    "translator": "llama_1b",
    "coder": "qwen3.5-2b",
}

TASKS = {
    "single-load": {
        "requests": [{"model_type": "chatbot", "complexity": "standard"}] * 9
    },
    "multi-load": {
        "requests": [
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "coder", "complexity": "reasoning"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "reasoning"},
        ]
    },
    "quality-limit": {
        "requests": [
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "chatbot", "complexity": "reasoning"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "reasoning"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "coder", "complexity": "reasoning"},
        ]
    },
    "ram-pressure": {
        "requests": [
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "translator", "complexity": "standard"},
            {"model_type": "coder", "complexity": "reasoning"},
            {"model_type": "chatbot", "complexity": "reasoning"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "coder", "complexity": "reasoning"},
            {"model_type": "chatbot", "complexity": "standard"},
            {"model_type": "translator", "complexity": "reasoning"},
            {"model_type": "coder", "complexity": "standard"},
            {"model_type": "chatbot", "complexity": "reasoning"},
            {"model_type": "translator", "complexity": "standard"},
        ]
    },
}