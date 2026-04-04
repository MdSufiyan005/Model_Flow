---
title: ModelFlow LLM Orchestrator
emoji: 🔊
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---

https://haider584-modelflow.hf.space

# ModelFlow - LLM Infrastructure Orchestrator

## Environment Description & Motivation

ModelFlow is a high-fidelity OpenEnv simulation designed for real-world LLM inference server management. The motivation behind this environment is to simulate the complexities of managing a constrained CPU-based inference cluster where dynamic memory spikes, quantization tradeoffs, and hardware limits are present. 

An AI agent must act as an infrastructure orchestrator to balance RAM capacity, quantization tiers, model loads, and CPU contention. The goal is to clear an incoming request queue accurately and efficiently while avoiding Out-Of-Memory (OOM) errors and time penalties.

## Action Space

The environment uses a discrete, parameterized action space:

- **`LOAD(model_id, quant_type)`**: Load a model to RAM. Incurs a time penalty proportional to the model's weight size and load-time metrics.
- **`EXECUTE(model_id, quant_type, batch_size)`**: Run inference for a batch of queued requests matching the model. Execution takes multiple steps and incurs a dynamic RAM peak.
- **`EVICT(model_id, quant_type)`**: Remove a resident model to free up RAM.
- **`REPLACE(model_id, quant_type, evict_model_id, evict_quant_type)`**: Swap out one model for another in a single fluid operation.
- **`IDLE`**: Wait one step. Incurs a penalty for queue aging.

## Observation Space

Agents receive dynamic state feedback via a Pydantic model (`ModelFlowObservation`):

- **`ram_used_mb`**: Current allocated RAM (weights + runtime overhead).
- **`ram_limit_mb`**: Physical RAM capacity limit.
- **`pressure_spike_mb`**: Random external memory pressure simulating other system processes consuming RAM.
- **`queue`**: List of pending requests (`RequestInfo`) mapped to roles, varying in complexity (standard vs. reasoning).
- **`loaded_models`**: Dictionary of currently resident models with size, tier, and live performance statistics.
- **`model_summary`**: Catalog of available model performance specifications.

## Task Descriptions & Expected Difficulty

ModelFlow includes 4 carefully calibrated tasks, each governed by programmatic graders (0.0 - 1.0 scale):

1. **Multi Load (Easy)**: 12 mixed requests. Tests the agent's ability to carefully pack all 3 model families (Gemma, Llama, Qwen) into RAM simultaneously without triggering OOM.
2. **Single Load (Medium)**: 9 homogeneous requests. Penalizes re-loading/thrashing; requires stable model residency over consecutive requests.
3. **Quality Limit (Hard)**: 14 mixed requests heavily reliant on reasoning tasks. Demands careful selection of high-tier quantization models while staying under tighter RAM limits.
4. **RAM Pressure (Extreme)**: 12 complex requests during heavy random spike conditions with tighter baseline limits. Deeply tests safety margins, proactive evictions, and OOM avoidance strategies.

## Setup & Usage Instructions

### Prerequisites

- `openenv-core`
- `groq` (Optional for remote inference proxy testing)
- `openai` (For the `inference.py` LLM-driven baseline agent)

### Local Validation

```bash
# Set up dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the baseline agent orchestrator 
python inference.py

# Run the OpenEnv interactive server interface (optional)
uvicorn model_flow.server.app:app --port 8000
```

### Docker 

To run the orchestrator environment in a container:

```bash
# Build the image from project root
docker build -t modelflow_test -f model_flow/Dockerfile model_flow/ or
docker build -t modelflow_test -f Dockerfile .

# Run the container (Map host:container port 8000)
docker run -d --name openenv_cont -p 8000:8000 modelflow_test

# Access the interface
# http://localhost:8000/docs      <- API
# http://localhost:8000/dashboard <- Retro Dashboard
```

## Baseline Scores

The baseline agent orchestrator, leveraging heuristics and safety overrides, achieves the following calibrated scores out of 1.0. These results were obtained using the following agent configuration:

- **LLM Agent**: `llama-3.3-70b-versatile`
- **Inference API**: **Groq API** (used for testing and development)

### Task Scores
- **Single Load**: 1.000
- **Multi Load**: 0.800
- **Quality Limit**: 0.900
- **RAM Pressure**: 1.000

## Dataset

I recorded the benchmark data profiling on a **Intel i3 laptop with 8 GB RAM**.  
This real-world, low-resource profiling ensures the simulation accurately reflects practical constraints of edge and constrained inference orchestration.

**Benchmarking Code**:  
[https://github.com/MdSufiyan005/BenchMarking](https://github.com/MdSufiyan005/BenchMarking)

**Dataset Location**:  
`model_flow/Data/combined_model_metrics.json`

This JSON contains metrics including weight size, host memory, context/compute/KV overhead, load times, and average CPU usage for (`gemma-3-4b`, `llama_1b`, `qwen3.5-2b`). It is loaded automatically by `ModelFlowEnvironment` at initialization.