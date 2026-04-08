import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


USE_GROQ_ONLY = env_bool("USE_GROQ_ONLY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if USE_GROQ_ONLY:
    if not GROQ_API_KEY:
        raise ValueError("USE_GROQ_ONLY=1 but GROQ_API_KEY is missing")
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    active_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
else:
    if not API_KEY:
        raise ValueError("OpenAI/HF branch selected but HF_TOKEN/API_KEY is missing")
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    active_model = MODEL_NAME


BENCHMARK = "modelflow"
TASKS = ["single-load", "multi-load", "quality-limit", "ram-pressure"]
MAX_STEPS_PER_TASK = 30
TEMPERATURE = 0.1
MAX_TOKENS = 600
CONTEXT_HISTORY_STEPS = 6
MAX_RETRIES = 4
BASE_BACKOFF_S = 2.0
SYSTEM_OVERHEAD_MB = 1100

ROLE_TO_MODEL: Dict[str, str] = {
    "chatbot": "gemma-3-4b",
    "translator": "llama_1b",
    "coder": "qwen3.5-2b",
}

QUANT_TIER: Dict[str, str] = {
    "Q4_K_M": "low",
    "Q5_K_M": "medium",
    "Q6_K": "high",
    "Q8_0": "risky",
}

REASONING_MIN_QUANT = "Q6_K"
REASONING_QUANTS = {"Q6_K", "Q8_0"}