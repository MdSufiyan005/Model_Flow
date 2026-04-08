def compress_step(step_num, action, reward, feedback, error) -> str:
    mod = action.model_id or ""
    quant = action.quant_type or ""
    result = f"ERR:{error[:80]}" if error else (feedback[:80] if feedback else "OK")
    return f"S{step_num}: {action.command}({mod}-{quant}) b={action.batch_size} R{reward:+.1f} | {result}"
    