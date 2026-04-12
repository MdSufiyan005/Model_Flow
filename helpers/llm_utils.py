# """
# helpers/llm_utils.py

# Thin wrapper around the OpenAI client.
# The client is owned by inference.py (hackathon requirement); we import it
# from there.  All other call parameters come from config.
# """

# from typing import Dict, List

# from config import active_model, TEMPERATURE, MAX_TOKENS


# def llm_call(messages: List[Dict]) -> str:
#     """
#     Single-shot LLM call that returns the raw text response.

#     Imports the client from inference at call time to avoid a circular import
#     at module load (inference.py imports helpers, helpers imports inference).
#     """
#     import inference as _inf  # late import — safe because inference is already loaded

#     response = _inf.client.chat.completions.create(
#         model=active_model,
#         messages=messages,
#         temperature=TEMPERATURE,
#         max_tokens=MAX_TOKENS,
#     )
#     return (response.choices[0].message.content or "").strip()