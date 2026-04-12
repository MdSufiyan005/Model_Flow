---
title: ModelFlow LLM Orchestrator
emoji: ⚙️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# ModelFlow ⚙️

**ModelFlow** is a reinforcement-learning environment and benchmark for LLM inference scheduling on RAM-constrained systems. At each step the agent observes the current system state - what models are loaded, how much RAM is free, what requests are queued and issues a single scheduling command. The goal is to serve every request in the queue as fast as possible while avoiding out-of-memory errors, quality degradation, and unnecessary model churn.

**Live demo:** [huggingface.co/spaces/MdSufiyan005/modelflow](https://huggingface.co/spaces/MdSufiyan005/modelflow)

(Interactive dashboard for manual exploration and debugging. The environment fully implements the OpenEnv API (step(), reset(), state()), enabling external agents to interact programmatically. The UI does not include a built-in agent.)


## Dashboard

![Dashboard 1](Images/dash1.png)
![Dashboard 2](Images/dash2.png)


## Quick Setup

### Local
```bash
uv venv && source .venv/bin/activate
uv sync

export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py                                    # run the agent
python -m uvicorn server.app:app --port 8000           # start the dashboard
```

### Docker
```bash
docker build -t modelflow -f Dockerfile .
docker run -d -p 8000:8000 modelflow
```

**Dashboard:** http://localhost:8000/  
**API docs:** http://localhost:8000/docs

## Project Structure
```text
model_flow/
├── inference.py                 # Agent loop, LLM calls, policy filter, episode logger
├── prompt.py                    # Observation → text, system prompt, decision tree, lessons
├── graders.py                   # Per-task scoring functions and sub-score helpers
├── models.py                    # Pydantic types: ModelFlowAction, RequestInfo, ModelFlowObservation
├── rewards.py                   # All per-step reward/penalty functions
├── config.py                    # API settings, constants, task definitions
├── openenv.yaml                 # OpenEnv configuration
├── Dockerfile
├── requirements.txt
│
├── server/
│   ├── modelflow_environment.py # Core simulation: RAM, queue, spikes, heat, SLA, demand shift
│   ├── app.py                   # FastAPI dashboard backend
│   ├── metrics_loader.py        # Parses benchmark JSON into the flat roster dict
│   └── constants.py             # Quant tiers, SLA values, heat thresholds, task configs
│
├── helpers/
│   ├── queue_utils.py           # Queue statistics and action parsing
│   ├── samplers.py              # Stochastic samplers: load time, host MB, quality failure
│   └── heat.py                  # Heat → bucket label converter
│
├── Data/
│   └── combined_model_metrics.json  # Real profiling data (RAM, latency, BLEU, ROUGE-L, perplexity)
│
├── scripts/                     # React dashboard frontend
│   ├── dashboard.html
│   ├── app.jsx
│   ├── index.jsx
│   ├── core.jsx
│   └── ui.jsx
│
└── Images/
```

## Baseline Scores (LLaMA 3.3 70B via Groq)
| Task          | Score |
|---------------|-------|
| Single Load   | 0.91  |
| Multi Load    | 0.88  |
| Quality Limit | 0.78  |
| RAM Pressure  | 0.67  |

## Dataset
Model metrics were profiled on an Intel i3 laptop with 8 GB RAM across 12 model/quant combinations (3 models × 4 quants). Measurements include host RAM range, load time range and variance, generation TPS, prompt TPS, CPU utilisation, BLEU, ROUGE-L, and perplexity.

**Source:** [github.com/MdSufiyan005/BenchMarking](https://github.com/MdSufiyan005/BenchMarking)

---

**For detailed information** (system architecture, request pipeline, RAM accounting, stochastic elements, agent actions, policy filter, quantization tiers, reward structure, full task descriptions & scoring, cross-episode learning, decision tree, tick caps, etc.) please refer to the `docs/` folder:

- [How It Works – System Architecture, Request Pipeline, RAM Accounting & Stochastic Elements](docs/how-it-works.md)
- [Agent Actions, Policy Filter & Quantization Tiers](docs/agent-actions.md)
- [Reward Structure & Tasks](docs/rewards-tasks.md)
- [Cross-Episode Learning, Decision Tree & Tick Caps](docs/advanced-features.md)
