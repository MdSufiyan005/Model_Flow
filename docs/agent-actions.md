# Agent Actions, Policy Filter & Quantization Tiers

## Agent Actions
Command | Required Fields | Description
---|---|---
LOAD | model_id, quant_type | Load a model into RAM. Takes 1–4 clock ticks (capped). Supports warm-cache fast-reload if recently evicted.
EXECUTE | model_id, quant_type, batch_size | Serve up to 8 matching requests. Takes 1–8 clock ticks (capped).
EVICT | model_id, quant_type (optional) | Unload a model and free its RAM. Adds it to the warm-cache.
REPLACE | evict_model_id/quant, model_id/quant | Atomic evict + load in a single step. Costs one extra clock tick.
DEFER | model_id (model_type to defer) | Move the front-of-queue request to a deferred sub-queue. Adds 3 age steps immediately. Deferred requests re-enter the queue before the next EXECUTE.
IDLE | — | Do nothing. Costs −15. Last resort only.

## Policy Filter
Before any action reaches the environment, _policy_filter() in inference.py intercepts five classes of invalid move:

1. Missing model_id on LOAD/EXECUTE/REPLACE → redirected to IDLE.
2. LOAD of an already-loaded key → redirected to EXECUTE with that key.
3. EXECUTE with a quant not matching the loaded slot → corrected to the actually-loaded quant, or passed through (to collect the real penalty) if the model is not loaded at all.
4. REPLACE where the new key is already loaded → redirected to EXECUTE (skips the destructive evict).
5. Self-replace (evict_key == new_key) → EXECUTE if loaded, IDLE if not.

These intercepts are logged and surfaced in the next observation so the agent can learn from them.

## Quantization Tiers
All model metrics are sourced from real profiling runs on an Intel i3 / 8 GB RAM machine.

Quant | Tier | Tier Rank | Complexity Supported | Notes
---|---|---|---|---
Q4_K_M | low | 0 | standard only | Smallest RAM footprint; cannot serve reasoning requests
Q5_K_M | medium | 1 | standard only | Moderate RAM; still below reasoning threshold
Q6_K | high | 2 | standard + reasoning | Recommended for mixed queues
Q8_0 | risky | 2 | standard + reasoning | Highest quality; largest KV-cache risk

The prompt layer computes an exec-safety flag for each candidate quant: if post-load free RAM < 2 000 MB (the EXEC_SAFETY_BUFFER_MB), the quant is flagged exec_safe=False and the prompt recommends a lower safe_quant instead. This is the primary mechanism for preventing RUNTIME OOMs before they happen.
