# How It Works

## System Architecture
The System consists of three tightly coupled layers:

1. Environment (server/modelflow_environment.py) — simulates a single host with 8 000 MB RAM (6 200 MB on hard/extreme tasks). It maintains loaded models, a request queue, a deferred sub-queue, RAM accounting, stochastic RAM spikes, per-model heat state, SLA tightening, and demand-shift injection. Every call to step() advances one logical clock tick and returns a new ModelFlowObservation.

2. Agent loop (inference.py) — calls an LLM (default: Qwen/Qwen2.5-72B-Instruct via the HuggingFace router) once per step. The raw JSON action from the model is passed through a deterministic policy filter that intercepts logically impossible or self-defeating moves before they reach the environment (see §Policy Filter below). Episode telemetry is written to episode_log.jsonl for cross-episode learning.

3. Prompt layer (prompt.py) — converts the observation into a structured text block containing: RAM status, loaded models, a per-model hint table (recommended quant, exec-safety flag, estimated net reward), queue breakdown, within-episode memory, a loop detector, and a five-step decision tree the model is instructed to follow. Past-episode lessons are injected at the top, filtered to the current task to avoid cross-task noise.

## Request Pipeline
Each request carries a model_type (chatbot, translator, coder), a complexity (standard or reasoning), and an age_steps counter that increments on every clock tick. The environment maps model_type → model_id via a fixed role table:
Role Model chatbot gemma-3-4b translator llama_1b coder qwen3.5-2b
An EXECUTE action succeeds only when the loaded quant's tier rank meets or exceeds the complexity minimum rank (reasoning requires tier high or risky, i.e. Q6_K or Q8_0).

## RAM Accounting
Every loaded model occupies host_mb bytes (sampled stochastically from a real-measurement range). During EXECUTE, the environment adds a dynamic KV-cache footprint on top:

```
total_peak = ram_used + active_spike_mb + SYSTEM_OVERHEAD_MB (1 100 MB) + ctx_mb + comp_mb + kv_mb_max × (batch / 8)
```

If total_peak > HARDWARE_RAM_MB the EXECUTE fails with a RUNTIME OOM penalty of −50. This is distinct from a load-time OOM (−30), which fires when the static model size alone exceeds available RAM.

## Stochastic Elements
The simulation introduces four sources of non-stationarity:

* Load-time jitter — actual load time is sampled from the real measurement range [load_range_ms.lo, load_range_ms.hi], not just the average.

* Host-size jitter — actual RAM footprint is sampled from [host_mb_range.lo, host_mb_range.hi].

* RAM pressure spikes — random spikes of 500–2 000 MB (up to 2 500 MB on hard tasks) last 3–10 steps with probability 0.10–0.18 per step.

* Model heat — each time a (model_id, quant) key is loaded, its heat counter increments. Quality-failure probability = min(0.07 × heat, 0.55). A quality failure halves the effective quality factor in the reward and applies an additional −4 penalty.
