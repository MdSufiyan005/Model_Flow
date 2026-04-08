from typing import Dict, List

from config import USE_GROQ_ONLY, active_model, client, TEMPERATURE, MAX_TOKENS


def llm_call(messages: List[Dict]) -> str:
    """Single-shot LLM call that returns the raw text response."""
    kwargs: Dict = dict(
        model=active_model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    if USE_GROQ_ONLY:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()