import sys

def print_visualization(task_name, step_num, obs, action=None, reward=0.0):
    width = 90
    border = "=" * width
    print("\n" + border, file=sys.stderr)
    label = "INITIAL STATE" if step_num == 0 else f"STEP {step_num:2d}"
    print(f" TASK: {task_name.upper():^20} | {label} ".center(width, "="), file=sys.stderr)
    print(border, file=sys.stderr)

    if obs.loaded_models:
        total_mb = sum(v["size_mb"] for v in obs.loaded_models.values())
        print(f" LOADED MODELS : ({len(obs.loaded_models)} slots, {total_mb}MB total)", file=sys.stderr)
        for key, slot in obs.loaded_models.items():
            print(
                f" → {key:<30} | tier={slot['tier']:<6} | "
                f"{slot['size_mb']}MB | {slot.get('gen_tps', 0):.1f}t/s",
                file=sys.stderr
            )
    else:
        print(" LOADED MODELS : NONE", file=sys.stderr)

    ram_pct = min(100, int(obs.ram_used_mb / obs.ram_limit_mb * 100))
    bar = "█" * (ram_pct // 5) + "░" * (20 - ram_pct // 5)
    print(f" RAM USAGE     : {obs.ram_used_mb:5d} / {obs.ram_limit_mb} MB [{bar}] {ram_pct:3d}%", file=sys.stderr)

    if obs.pressure_spike_mb > 0:
        print(
            f" MEMORY SPIKE  : {obs.pressure_spike_mb}MB active, "
            f"{obs.spike_steps_remaining} steps remaining",
            file=sys.stderr
        )

    if step_num == 0:
        print(" ACTION TAKEN  : ENVIRONMENT RESET (no action yet)", file=sys.stderr)
    else:
        mod = action.model_id or ""
        quant = action.quant_type or ""
        act_str = (
            f"{action.command}({mod}-{quant}, batch={action.batch_size})"
            if mod
            else f"{action.command}(batch={action.batch_size})"
        )
        print(f" ACTION TAKEN  : {act_str}", file=sys.stderr)

    print(f" REWARD        : {reward:+8.2f}", file=sys.stderr)

    if obs.last_action_error:
        print(f" ERROR         : {obs.last_action_error}", file=sys.stderr)
    elif obs.last_action_feedback:
        print(f" FEEDBACK      : {obs.last_action_feedback}", file=sys.stderr)

    print(f"\n QUEUE STATUS (pending: {len(obs.queue)})", file=sys.stderr)
    for i, req in enumerate(obs.queue[:12]):
        print(
            f" {i+1:2d}. {req.request_id:>8} | {req.model_type.upper().ljust(10)} | "
            f"{req.complexity.upper().ljust(10)} | age={req.age_steps:2d}",
            file=sys.stderr
        )

    if len(obs.queue) > 12:
        print(f" ... +{len(obs.queue) - 12} more requests", file=sys.stderr)

    print(
        f" PROGRESS      : {obs.info.get('completed', 0)} completed | "
        f"{len(obs.queue)} pending | Step: {step_num}",
        file=sys.stderr
    )
    print(border + "\n", file=sys.stderr)