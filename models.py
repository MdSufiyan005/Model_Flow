from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

Command = Literal["LOAD", "EXECUTE", "EVICT", "IDLE", "REPLACE"]
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

    # Multi-model co-loading:
    #   key   = "model_id-quant_type"  e.g. "llama_1b-Q4_K_M"
    #   value = {"model": str, "quant": str, "tier": str, "size_mb": int}
    loaded_models: Dict[str, dict] = Field(default_factory=dict)

    queue: List[RequestInfo] = Field(default_factory=list)
    available_model_types: List[str] = Field(
        default_factory=lambda: ["chatbot", "translator", "coder"]
    )
    # model_summary[role] = {"model_id": str, "stats": {quant: {size_mb, load_time_s, tier}}}
    model_summary: Dict[str, Dict] = Field(default_factory=dict)

    last_action_feedback: Optional[str] = None
    step_count: int = 0
    done: bool = False
    last_action_error: Optional[str] = None

    # Observable memory-pressure spike state
    pressure_spike_mb: int = 0
    spike_steps_remaining: int = 0

    info: Dict = Field(default_factory=dict)
    reward: float = 0.0