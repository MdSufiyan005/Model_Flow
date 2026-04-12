
# Reward Structure & Tasks

## Reward Structure
Rewards are shaped per-step and summed across the episode. Key values:

Event | Reward
---|---
Episode success (queue cleared) | +50
Episode timeout | −50
EXECUTE success (per request, standard) | up to +20 × quality_factor × tier_multiplier
EXECUTE success (per request, reasoning) | up to +30 × quality_factor × (1 − quant_penalty)
SLA overage (per request) | up to −40% of that request's gain
Quality degradation (heat-induced) | −4 flat + halved quality_factor
Deferred request served later | +5 to +9 bonus (age-scaled)
LOAD success | −load_time_s + up to +4 if ≥2 models loaded
LOAD already-loaded | −5
LOAD OOM | −30
REPLACE evict component (needed model) | −6
REPLACE evict component (unneeded model) | +2
REPLACE load success | +5 − load_time_s
EXECUTE not loaded | −10
EXECUTE RUNTIME OOM | −50
EXECUTE no-match (escalating) | −5 × fail_count², capped −45
EVICT success (unneeded model) | −2 net (−6 + 4)
EVICT success (still-needed model) | −6
IDLE (with pending requests) | −15
Clock tick penalty (per aged request) | up to −0.15 per request per tick

Reasoning requests additionally carry a REASONING_QUANT_PENALTY that scales the effective quality factor: Q4_K_M loses 35%, Q5_K_M 18%, Q6_K 6%, Q8_0 0%.

## Tasks
Four tasks of increasing difficulty. All are scored on a 0–0.99 scale by graders.py.

**Single Load (Easy)**  
Nine identical chatbot/standard requests. Tests whether the agent loads once, executes to completion, and avoids unnecessary churn. Scored primarily on completion (40%) and stability (22%).

**Multi Load (Medium)**  
Twelve mixed requests across all three model types, including reasoning. A demand shift fires at a randomly sampled step T ∈ [5, 8]: three additional requests of a different type are injected into the queue, and a shift_detected hint appears two steps later. Tests quant selection, demand adaptation, and step efficiency. Scored on completion (28%), reasoning completion (18%), and step efficiency (14%).

**Quality Limit (Hard)**  
Fourteen requests with sla_tightening=True: the SLA window shrinks by 2 steps every 4 steps (floor: 10 steps), starting at 40 steps. Tests heat management, quality-aware quant selection, and strategic deferral under a shrinking deadline. OOM errors apply a hard score ceiling via _oom_ceiling() (one OOM caps the score at ~0.62). Scored on completion (25%), quality accuracy (22%), and reasoning completion (18%).

**RAM Pressure (Extreme)**  
Twelve reasoning-heavy requests with both sla_tightening=True and demand_shift=True, on reduced hardware (6 200 MB RAM) with higher spike probability (0.18). Tests compound pressure: OOM avoidance + heat management + churn minimisation + strategic deferral. Three structural score modifiers apply on top of the weighted sub-scores:

* OOM ceiling: _oom_ceiling(oom_errors) — 0 OOMs → no cap; 1 OOM → max ~0.62; 2 OOMs → max ~0.41.
* Queue-not-cleared multiplier: ×0.75 if any request was abandoned.
* Step soft multiplier: kicks in above 16 steps, −4% per extra step.

Scored on completion (22%), quality accuracy (20%), and churn (18%).

