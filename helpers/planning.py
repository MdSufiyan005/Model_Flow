# """
# helpers/planning.py — v5

# Critical fixes based on OOM-CHECK trace (steps 2 and 9 were "no matching requests"
# penalties, NOT RAM OOMs):

#   - _has_servable_queue + _servable_batch_size: now mirror the environment's exact
#     EXECUTE condition (model mapping AND tier_rank >= complexity_min_rank).
#     This eliminates the -790 / -1062 penalties when the loaded model cannot serve
#     any queued request due to tier mismatch.

#   - IDLE→EXECUTE: verifies _has_servable_queue before issuing EXECUTE and caps
#     batch_size via _servable_batch_size (never sends batch=5 when only 2 requests
#     are actually servable).

#   - REPLACE churn fix: if the model being evicted STILL has servable requests,
#     EXECUTE it first instead of swapping (stops the llama↔qwen↔llama↔qwen loop).

#   - IDLE→LOAD: _base_free_mb completely removed. _best_loadable_model now uses
#     only _effective_free_mb (with spike) and selects the model with the MOST
#     pending requests that actually fits — not the oldest request's model (which
#     could be too large and trigger the 3-6 OOM loop).

#   - LOAD and runtime-OOM paths already hardened in v4; this version builds on them.
# """

# from __future__ import annotations
# from typing import TYPE_CHECKING, Dict, Optional, Tuple

# if TYPE_CHECKING:
#     from models import ModelFlowAction, ModelFlowObservation

# _TIER_RANK: Dict[str, int] = {
#     "low": 1, "medium": 2, "high": 3, "risky": 4,
# }
# _COMPLEXITY_MIN_RANK: Dict[str, int] = {
#     "simple": 1, "standard": 1, "reasoning": 3,
# }
# _REASONING_QUANTS         = {"Q6_K", "Q8_0"}
# _DEFAULT_REASONING_QUANT  = "Q6_K"
# _DEFAULT_SIMPLE_QUANT     = "Q4_K_M"
# _SYSTEM_OVERHEAD_MB       = 1100
# _QUANT_ORDER              = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


# # ── Helpers ──────────────────────────────────────────────────────────────────

# def _role_to_model(obs: "ModelFlowObservation") -> Dict[str, str]:
#     return {role: info["model_id"] for role, info in obs.model_summary.items()}


# def _needs_reasoning(obs: "ModelFlowObservation", model_id: str) -> bool:
#     rtm = _role_to_model(obs)
#     return any(
#         rtm.get(req.model_type) == model_id and req.complexity == "reasoning"
#         for req in obs.queue
#     )


# def _has_servable_queue(obs: "ModelFlowObservation", model_id: str) -> bool:
#     """True if ANY queued request can be served by this loaded model
#     (model mapping + tier sufficient). Mirrors the environment's EXECUTE guard."""
#     key = _loaded_key(obs, model_id)
#     if not key:
#         return False
#     slot = obs.loaded_models[key]
#     tier = slot.get("tier", "low")
#     tier_rank = _TIER_RANK.get(tier, 1)

#     rtm = _role_to_model(obs)
#     for req in obs.queue:
#         if rtm.get(req.model_type) == model_id:
#             min_rank = _COMPLEXITY_MIN_RANK.get(req.complexity, 1)
#             if tier_rank >= min_rank:
#                 return True
#     return False


# def _servable_batch_size(obs: "ModelFlowObservation", model_id: str) -> int:
#     """Number of queued requests (≤8) that this loaded model can actually serve.
#     Used so EXECUTE never receives a batch_size larger than what the environment
#     will accept."""
#     key = _loaded_key(obs, model_id)
#     if not key:
#         return 0
#     slot = obs.loaded_models[key]
#     tier = slot.get("tier", "low")
#     tier_rank = _TIER_RANK.get(tier, 1)

#     rtm = _role_to_model(obs)
#     count = 0
#     for req in obs.queue:
#         if rtm.get(req.model_type) == model_id:
#             min_rank = _COMPLEXITY_MIN_RANK.get(req.complexity, 1)
#             if tier_rank >= min_rank:
#                 count += 1
#                 if count == 8:
#                     break
#     return count


# def _best_loadable_model(obs: "ModelFlowObservation") -> Optional[Tuple[str, str]]:
#     """Selects the model with the MOST pending requests that fits entirely in
#     _effective_free_mb (including spike). Prevents the "oldest-request model is
#     too large" OOM loop."""
#     eff_free = _effective_free_mb(obs)
#     if eff_free <= 0:
#         return None

#     rtm = _role_to_model(obs)
#     model_pending: Dict[str, int] = {}
#     for req in obs.queue:
#         mid = rtm.get(req.model_type)
#         if mid:
#             model_pending[mid] = model_pending.get(mid, 0) + 1

#     candidates = []
#     for mid, count in model_pending.items():
#         if _loaded_key(obs, mid):
#             continue
#         needed_quant = _best_quant(obs, mid)
#         size = _model_size_mb(obs, mid, needed_quant)
#         if size <= eff_free and size < 9999:
#             candidates.append((count, -size, mid, needed_quant))

#     if not candidates:
#         return None

#     candidates.sort(reverse=True)
#     _, _, best_mid, best_quant = candidates[0]
#     return best_mid, best_quant


# def _best_quant(obs: "ModelFlowObservation", model_id: str) -> str:
#     return _DEFAULT_REASONING_QUANT if _needs_reasoning(obs, model_id) else _DEFAULT_SIMPLE_QUANT


# def _model_size_mb(obs: "ModelFlowObservation", model_id: str, quant: str) -> int:
#     for info in obs.model_summary.values():
#         if info["model_id"] == model_id:
#             return info["stats"].get(quant, {}).get("size_mb", 9999)
#     return 9999


# def _effective_free_mb(obs: "ModelFlowObservation") -> int:
#     """Conservative: subtracts spike. Negative means spike has eaten all headroom."""
#     return int(
#         obs.ram_limit_mb - obs.ram_used_mb - obs.pressure_spike_mb - _SYSTEM_OVERHEAD_MB
#     )


# def _loaded_key(obs: "ModelFlowObservation", model_id: str) -> Optional[str]:
#     for key, slot in obs.loaded_models.items():
#         if slot["model"] == model_id:
#             return key
#     return None


# def _quant_from_key(key: str, obs: "ModelFlowObservation") -> str:
#     return obs.loaded_models.get(key, {}).get("quant", key.split("-")[-1])


# def _runtime_oom(obs: "ModelFlowObservation", key: str, batch_size: int) -> bool:
#     slot = obs.loaded_models.get(key)
#     if not slot:
#         return False

#     ctx_mb     = slot.get("ctx_mb",     0)
#     comp_mb    = slot.get("comp_mb",    0)
#     kv_mb_max  = slot.get("kv_mb_max",  0)

#     if ctx_mb == 0 and comp_mb == 0 and kv_mb_max == 0:
#         return _effective_free_mb(obs) < 0

#     kv_dynamic = kv_mb_max * (batch_size / 8.0)
#     total_peak = (
#         obs.ram_used_mb
#         + obs.pressure_spike_mb
#         + _SYSTEM_OVERHEAD_MB
#         + ctx_mb
#         + comp_mb
#         + kv_dynamic
#     )
#     return total_peak > obs.ram_limit_mb


# def _safe_batch_size(obs: "ModelFlowObservation", key: str, desired: int) -> int:
#     slot = obs.loaded_models.get(key)
#     if not slot:
#         return 0

#     ctx_mb    = slot.get("ctx_mb",    0)
#     comp_mb   = slot.get("comp_mb",   0)
#     kv_mb_max = slot.get("kv_mb_max", 0)

#     if ctx_mb == 0 and comp_mb == 0 and kv_mb_max == 0:
#         return desired if _effective_free_mb(obs) >= 0 else 0

#     headroom = (
#         obs.ram_limit_mb
#         - obs.ram_used_mb
#         - obs.pressure_spike_mb
#         - _SYSTEM_OVERHEAD_MB
#         - ctx_mb
#         - comp_mb
#     )
#     if headroom <= 0 or kv_mb_max == 0:
#         return 0
#     max_batch = int(headroom / (kv_mb_max / 8.0))
#     return max(0, min(desired, max_batch))


# def _evict_least_needed(
#     obs: "ModelFlowObservation",
#     exclude_model_id: Optional[str] = None,
# ) -> Optional["ModelFlowAction"]:
#     rtm           = _role_to_model(obs)
#     needed_models = {rtm.get(req.model_type) for req in obs.queue}
#     candidates    = []
#     for key, slot in obs.loaded_models.items():
#         mid = slot["model"]
#         if mid == exclude_model_id:
#             continue
#         candidates.append((mid in needed_models, -slot.get("size_mb", 0), key, mid, slot.get("quant", "")))
#     if not candidates:
#         return None
#     candidates.sort()
#     _, _, _, mid, quant = candidates[0]
#     return _make("EVICT", model_id=mid, quant_type=quant)


# def _find_fitting_quant(
#     obs: "ModelFlowObservation",
#     model_id: str,
#     min_quant: str = "Q4_K_M",
#     free_override: Optional[int] = None,
# ) -> Optional[str]:
#     free    = free_override if free_override is not None else _effective_free_mb(obs)
#     min_idx = _QUANT_ORDER.index(min_quant) if min_quant in _QUANT_ORDER else 0
#     for quant in reversed(_QUANT_ORDER[min_idx:]):
#         size = _model_size_mb(obs, model_id, quant)
#         if size <= free and size < 9999:
#             return quant
#     return None


# def _make(command, model_id=None, quant_type=None, batch_size=8,
#           evict_model_id=None, evict_quant_type=None) -> "ModelFlowAction":
#     from models import ModelFlowAction
#     return ModelFlowAction(
#         command=command, model_id=model_id, quant_type=quant_type,
#         batch_size=batch_size, evict_model_id=evict_model_id,
#         evict_quant_type=evict_quant_type,
#     )


# # ── Public API ────────────────────────────────────────────────────────────────

# def apply_planning_override(
#     action: "ModelFlowAction",
#     obs:    "ModelFlowObservation",
# ) -> "ModelFlowAction":

#     cmd = action.command

#     # ── 1. LOAD ──────────────────────────────────────────────────────────────
#     if cmd == "LOAD" and action.model_id:
#         model_id = action.model_id
#         quant    = action.quant_type or _best_quant(obs, model_id)

#         if quant not in _REASONING_QUANTS and _needs_reasoning(obs, model_id):
#             quant = _DEFAULT_REASONING_QUANT

#         size = _model_size_mb(obs, model_id, quant)
#         free = _effective_free_mb(obs)

#         if size > free:
#             if obs.loaded_models:
#                 evict = _evict_least_needed(obs, exclude_model_id=model_id)
#                 if evict:
#                     evict_key = f"{evict.model_id}-{evict.quant_type}"
#                     freed     = obs.loaded_models.get(evict_key, {}).get("size_mb", 0)
#                     if size <= free + freed:
#                         return evict

#             min_q = _DEFAULT_REASONING_QUANT if _needs_reasoning(obs, model_id) else "Q4_K_M"
#             fq    = _find_fitting_quant(obs, model_id, min_quant=min_q)
#             if fq:
#                 return _make("LOAD", model_id=model_id, quant_type=fq,
#                              batch_size=action.batch_size or 8)

#             rtm         = _role_to_model(obs)
#             needed_mids = list({rtm.get(req.model_type) for req in obs.queue
#                                  if rtm.get(req.model_type) and rtm.get(req.model_type) != model_id})
#             for alt in needed_mids:
#                 aq = _best_quant(obs, alt)
#                 s  = _model_size_mb(obs, alt, aq)
#                 if s <= free and s < 9999:
#                     return _make("LOAD", model_id=alt, quant_type=aq,
#                                  batch_size=action.batch_size or 8)

#             evict = _evict_least_needed(obs)
#             if evict:
#                 return evict

#         return _make("LOAD", model_id=model_id, quant_type=quant,
#                      batch_size=action.batch_size or 8)

#     # ── 2. EXECUTE ───────────────────────────────────────────────────────────
#     if cmd == "EXECUTE" and action.model_id:
#         key  = _loaded_key(obs, action.model_id)
#         slot = obs.loaded_models.get(key or "", {})
#         tier = slot.get("tier", "low")

#         # 2a. No servable requests for this model → switch to a loaded model that has them
#         if not _has_servable_queue(obs, action.model_id):
#             rtm = _role_to_model(obs)
#             for req in obs.queue:
#                 mid = rtm.get(req.model_type)
#                 if mid and _loaded_key(obs, mid) and _has_servable_queue(obs, mid):
#                     lk = _loaded_key(obs, mid)
#                     lq = _quant_from_key(lk, obs)
#                     servable_b = _servable_batch_size(obs, mid)
#                     if servable_b > 0:
#                         safe_b = _safe_batch_size(obs, lk, servable_b) if _runtime_oom(obs, lk, servable_b) else servable_b
#                         if safe_b > 0:
#                             return _make("EXECUTE", model_id=mid, quant_type=lq, batch_size=safe_b)
#             return _make("IDLE")

#         # 2b. Tier too low for reasoning → upgrade quant on same model
#         if (_needs_reasoning(obs, action.model_id) and
#                 _TIER_RANK.get(tier, 1) < _COMPLEXITY_MIN_RANK["reasoning"]):
#             return _make("REPLACE",
#                          model_id=action.model_id,
#                          quant_type=_DEFAULT_REASONING_QUANT,
#                          batch_size=action.batch_size or 8,
#                          evict_model_id=action.model_id,
#                          evict_quant_type=action.quant_type)

#         # 2c. Runtime OOM check — reduce batch or evict
#         if key and _runtime_oom(obs, key, action.batch_size or 8):
#             safe = _safe_batch_size(obs, key, action.batch_size or 8)
#             if safe > 0:
#                 return _make("EXECUTE", model_id=action.model_id,
#                              quant_type=action.quant_type, batch_size=safe)
#             evict = _evict_least_needed(obs, exclude_model_id=action.model_id)
#             if evict:
#                 return evict

#     # ── 3. REPLACE ───────────────────────────────────────────────────────────
#     if cmd == "REPLACE" and action.model_id and action.quant_type:
#         # REPLACE churn fix: if the model we are about to evict STILL has
#         # servable requests, EXECUTE it first (only for true model swaps)
#         if (action.evict_model_id and
#                 action.evict_model_id != action.model_id and
#                 _has_servable_queue(obs, action.evict_model_id)):
#             lk = _loaded_key(obs, action.evict_model_id)
#             if lk:
#                 lq = action.evict_quant_type or _quant_from_key(lk, obs)
#                 servable_b = _servable_batch_size(obs, action.evict_model_id)
#                 if servable_b > 0:
#                     safe_b = (_safe_batch_size(obs, lk, servable_b)
#                               if _runtime_oom(obs, lk, servable_b) else servable_b)
#                     if safe_b > 0:
#                         return _make("EXECUTE", model_id=action.evict_model_id,
#                                      quant_type=lq, batch_size=safe_b)

#         # Normal REPLACE sizing logic
#         evict_key  = (f"{action.evict_model_id}-{action.evict_quant_type}"
#                       if action.evict_model_id else None)
#         freed      = obs.loaded_models.get(evict_key, {}).get("size_mb", 0) if evict_key else 0
#         free_after = _effective_free_mb(obs) + freed
#         size       = _model_size_mb(obs, action.model_id, action.quant_type)

#         if size > free_after:
#             min_q = _DEFAULT_REASONING_QUANT if _needs_reasoning(obs, action.model_id) else "Q4_K_M"
#             fq    = _find_fitting_quant(obs, action.model_id, min_quant=min_q,
#                                         free_override=free_after)
#             if fq:
#                 return _make("REPLACE",
#                              model_id=action.model_id, quant_type=fq,
#                              batch_size=action.batch_size or 8,
#                              evict_model_id=action.evict_model_id,
#                              evict_quant_type=action.evict_quant_type)
#             if evict_key and evict_key in obs.loaded_models:
#                 return _make("EVICT", model_id=action.evict_model_id,
#                              quant_type=action.evict_quant_type)

#     # ── 4. IDLE with non-empty queue ─────────────────────────────────────────
#     if cmd == "IDLE" and obs.queue:
#         rtm = _role_to_model(obs)

#         # 4a. Prefer EXECUTE on the oldest request whose model is loaded AND servable
#         target_model: Optional[str] = None
#         for req in sorted(obs.queue, key=lambda r: r.age_steps, reverse=True):
#             mid = rtm.get(req.model_type)
#             if mid and _loaded_key(obs, mid) and _has_servable_queue(obs, mid):
#                 target_model = mid
#                 break

#         if target_model:
#             loaded_k = _loaded_key(obs, target_model)
#             lquant = _quant_from_key(loaded_k, obs)

#             if _needs_reasoning(obs, target_model) and lquant not in _REASONING_QUANTS:
#                 return _make("REPLACE",
#                              model_id=target_model,
#                              quant_type=_DEFAULT_REASONING_QUANT,
#                              batch_size=8,
#                              evict_model_id=target_model,
#                              evict_quant_type=lquant)

#             servable_b = _servable_batch_size(obs, target_model)
#             desired = servable_b
#             if desired > 0 and _runtime_oom(obs, loaded_k, desired):
#                 desired = _safe_batch_size(obs, loaded_k, servable_b)

#             if desired > 0:
#                 return _make("EXECUTE", model_id=target_model,
#                              quant_type=lquant, batch_size=desired)

#             # Even batch=1 OOMs — evict something else
#             evict = _evict_least_needed(obs, exclude_model_id=target_model)
#             if evict:
#                 return evict

#         # 4b. No usable loaded model → load the best-fitting model (most pending requests)
#         load_info = _best_loadable_model(obs)
#         if load_info:
#             target_model, needed_quant = load_info
#             return _make("LOAD", model_id=target_model,
#                          quant_type=needed_quant, batch_size=8)

#         # 4c. Nothing can be executed or loaded → evict least-needed model
#         evict = _evict_least_needed(obs)
#         if evict:
#             return evict

#     return action