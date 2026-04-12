from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

Command = Literal["LOAD", "EXECUTE", "EVICT", "IDLE", "REPLACE", "DEFER"]
Complexity = Literal["standard", "reasoning"]


class ModelFlowAction(BaseModel):
    command: Command
    model_id: Optional[str] = None
    quant_type: Optional[str] = None
    batch_size: Optional[int] = None
    evict_model_id: Optional[str] = None
    evict_quant_type: Optional[str] = None


class RequestInfo(BaseModel):
    request_id: str
    model_type: str           # "chatbot" | "translator" | "coder"
    complexity: Complexity = "standard"
    age_steps: int = 0
    prompt_tokens: int = 64
    gen_tokens: int = 128

    @property
    def reasoning(self) -> bool:
        return self.complexity == "reasoning"


class ModelFlowObservation(BaseModel):
    ram_used_mb: int
    ram_limit_mb: int = 8000
    loaded_models: Dict[str, dict] = Field(default_factory=dict)

    queue: List[RequestInfo] = Field(default_factory=list)
    available_model_types: List[str] = Field(
        default_factory=lambda: ["chatbot", "translator", "coder"]
    )
    # model_summary[role] = {"model_id": str, "stats": {quant: {size_mb, gen_tps, tier}}}
    model_summary: Dict[str, Dict] = Field(default_factory=dict)

    last_action_feedback: Optional[str] = None
    step_count: int = 0
    done: bool = False
    last_action_error: Optional[str] = None

    # Observable memory-pressure spike state
    pressure_spike_mb: int = 0
    spike_steps_remaining: int = 0

    # partial-observability signals
    # Bucketed heat per loaded model key: "low" | "medium" | "high"
    # Agent sees category, NOT the raw integer heat count.
    model_heat_signals: Dict[str, str] = Field(default_factory=dict)

    # Last up-to-3 execute outcomes: True = quality OK, False = quality degraded.
    # Oldest first. Empty until first EXECUTE.
    recent_quality_outcomes: List[bool] = Field(default_factory=list)

    # Appears as "shift_detected" after 2 steps of anomalous queue composition,
    # None otherwise. Agent must detect and adapt before this surfaces.
    demand_hint: Optional[str] = None

    # Current SLA window in steps. Tightens on hard/extreme tasks.
    # A request served with age_steps > current_sla_steps counts as late.
    current_sla_steps: int = 40

    # How many requests are currently sitting in the deferred sub-queue.
    deferred_count: int = 0

    info: Dict = Field(default_factory=dict)
    reward: float = 0.0