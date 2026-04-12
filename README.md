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

ModelFlow is a benchmark where an AI agent acts as an LLM inference scheduler. The agent manages a constrained RAM environment — loading, executing, and evicting models to serve a stream of incoming requests as efficiently as possible.

**Live demo:** [huggingface.co/spaces/MdSufiyan005/modelflow](https://huggingface.co/spaces/MdSufiyan005/modelflow)

---

## How It Works

Each episode, the agent receives an observation of the current system state and decides what action to take. The goal is to clear the request queue with minimal latency, zero OOM errors, and smart model reuse.

### Agent Actions

| Action | Description |
|---|---|
| `LOAD(model_id, quant_type)` | Load a model into RAM |
| `EXECUTE(model_id, quant_type, batch_size)` | Process matching queued requests |
| `EVICT(model_id, quant_type)` | Free RAM by unloading a model |
| `REPLACE(evict_model, load_model)` | Swap one model for another in a single step |
| `DEFER(model_type)` | Move a request to a deferred sub-queue to serve later |
| `IDLE` | Do nothing (penalised — last resort) |

### Quantization Tiers

Quant level determines model quality and RAM cost. Reasoning requests require `Q6_K` or higher.

| Quant | Tier | Use for |
|---|---|---|
| `Q4_K_M` | low | Standard requests only |
| `Q5_K_M` | medium | Standard requests |
| `Q6_K` | high | Standard + reasoning |
| `Q8_0` | risky | Standard + reasoning (highest quality, most RAM) |

---

## Tasks

Four tasks of increasing difficulty, each scored independently.

| Task | Difficulty | Description |
|---|---|---|
| **Single Load** | Easy | 9 identical chatbot requests — tests efficient model reuse |
| **Multi Load** | Medium | 12 mixed requests across 3 models — tests quant selection and demand adaptation |
| **Quality Limit** | Hard | 14 requests with tightening SLA — tests heat management and quality-aware scheduling |
| **RAM Pressure** | Extreme | 12 reasoning-heavy requests with RAM spikes and demand shifts — tests OOM avoidance under compound pressure |

### Baseline Scores (LLaMA 3.3 70B via Groq)

| Task | Score |
|---|---|
| Single Load | 0.88 |
| Multi Load | 0.64 |
| Quality Limit | 0.69 |
| RAM Pressure | 0.54 |

---

## Dashboard

![Dashboard 1](Images/dash1.png)
![Dashboard 2](Images/dash2.png)
![Terminal Output](Images/Terminal-Output_.png)

---

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

**Dashboard:** `http://localhost:8000/` · **API docs:** `http://localhost:8000/docs`

---

## Project Structure

```text
model_flow/
├── inference.py                 # Main agent loop, LLM calls, action parsing, logging
├── prompt.py                    # Builds prompts from environment observations
├── graders.py                   # Task scoring and evaluation logic
├── models.py                    # Shared data models and types
├── rewards.py                   # Reward calculation logic
├── config.py                    # Configuration, API settings, constants
├── openenv.yaml                 # OpenEnv configuration (environment setup)
├── Dockerfile                   # Containerization setup for deployment
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── server/
│   ├── modelflow_environment.py # Core simulation environment (RAM, queue, rewards)
│   ├── app.py                   # FastAPI backend server
│   ├── metrics_loader.py        # Loads benchmark/model metrics
│   └── constants.py             # Quant tiers, SLA values, task definitions
│
├── helpers/                     # Core utility modules
│   ├── queue_utils.py           # Queue handling and scheduling utilities
│
├── Data/
│   └── combined_model_metrics.json  # Benchmark dataset (RAM, latency, quality)
│
├── scripts/                     # Frontend (React dashboard)
│   ├── dashboard.html
│   ├── app.jsx
│   ├── index.jsx
│   ├── core.jsx
│   └── ui.jsx
│
└── Images/                      # Screenshots / UI previews
```


---

## Dataset

Model metrics profiled on an Intel i3 laptop (8 GB RAM).  
Source: [github.com/MdSufiyan005/BenchMarking](https://github.com/MdSufiyan005/BenchMarking)