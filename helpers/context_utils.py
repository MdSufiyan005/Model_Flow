# """
# helpers/context_utils.py

# Compresses a completed step into a short string for the LLM history window.

# Design: keep the error message long enough to contain MB numbers (the LLM
# needs "needs 3252MB, only 2872MB free" to avoid repeating the same OOM),
# but strip boilerplate from success feedback to save tokens.
# """


# def compress_step(step_num, action, reward, feedback, error) -> str:
#     mod   = action.model_id   or ""
#     quant = action.quant_type or ""

#     if error:
#         # Keep full OOM messages — the MB numbers prevent repeat mistakes
#         result = f"ERR:{error[:120]}"
#     elif feedback:
#         # Strip verbose parts of success messages, keep key numbers
#         fb = feedback
#         # "Loaded gemma-3-4b-Q6_K (cold) in 4.2s." → keep as-is, it's short
#         # "Executed batch 5 on gemma-3-4b-Q6_K in 12.3s (contention: 12%)." → shorten
#         if fb.startswith("Executed batch"):
#             # Extract just batch count and time
#             parts = fb.split()
#             try:
#                 batch = parts[2]
#                 time_idx = parts.index("in") + 1
#                 time_s   = parts[time_idx]
#                 result   = f"OK:batch={batch} t={time_s}"
#             except (ValueError, IndexError):
#                 result = fb[:60]
#         else:
#             result = fb[:80]
#     else:
#         result = "OK"

#     return (
#         f"S{step_num}:{action.command}({mod}-{quant})"
#         f" b={action.batch_size} R{reward:+.0f}|{result}"
#     )