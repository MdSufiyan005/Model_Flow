from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ---------------------------------------------------------------------
# Robust path handling
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if ROOT.name == "server":
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
from models import ModelFlowAction, ModelFlowObservation
from server.test_environment import ModelFlowEnvironment

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(
    title="ModelFlow",
    description="FastAPI environment server for ModelFlow",
    version="1.0.0",
)

SCRIPTS_PATH = ROOT / "scripts"
SCRIPTS_PATH.mkdir(parents=True, exist_ok=True)

app.mount("/scripts", StaticFiles(directory=str(SCRIPTS_PATH)), name="scripts")

singleton_env = ModelFlowEnvironment()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    return jsonable_encoder(obj, by_alias=True)


def _schema_for(model_cls: Any) -> Dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    if hasattr(model_cls, "schema"):
        return model_cls.schema()
    return {}


def _current_observation() -> Any:
    if hasattr(singleton_env, "_get_observation"):
        return singleton_env._get_observation()
    if hasattr(singleton_env, "observation"):
        return singleton_env.observation
    return None


def _build_action_payload(action: Dict[str, Any]) -> Dict[str, Any]:
    cmd = (action or {}).get("command", "IDLE")
    payload: Dict[str, Any] = {"command": cmd}

    if cmd == "IDLE":
        return payload

    if cmd in {"LOAD", "EXECUTE", "EVICT", "REPLACE"}:
        if action.get("model_id"):
            payload["model_id"] = action["model_id"]
        if action.get("quant_type"):
            payload["quant_type"] = action["quant_type"]

    if cmd == "EXECUTE":
        batch_size = action.get("batch_size", 1)
        try:
            payload["batch_size"] = int(batch_size)
        except Exception:
            payload["batch_size"] = 1

    if cmd == "REPLACE":
        if action.get("evict_model_id"):
            payload["evict_model_id"] = action["evict_model_id"]
        if action.get("evict_quant_type"):
            payload["evict_quant_type"] = action["evict_quant_type"]

    return payload


def _read_dashboard_html() -> str:
    path = SCRIPTS_PATH / "dashboard.html"
    if path.exists():
        return path.read_text(encoding="utf-8")

    return """<!DOCTYPE html>
<html>
  <head><meta charset="utf-8"><title>ModelFlow</title></head>
  <body style="background:#070502;color:#fff;font-family:sans-serif;padding:24px">
    <h1>Dashboard missing</h1>
    <p>scripts/dashboard.html was not found.</p>
  </body>
</html>"""


def _coerce_reset_output(obs: Any, task_name: str) -> Dict[str, Any]:
    obs_json = _to_jsonable(obs)
    return {
        "observation": obs_json,
        "state": {
            "episode_id": "singleton",
            "step_count": getattr(obs, "step_count", obs_json.get("step_count", 0) if isinstance(obs_json, dict) else 0),
            "task_name": task_name,
        },
    }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(_read_dashboard_html())

@app.get("/web")
async def root():
    return {"message": "ModelFlow Orchestrator is running!"}

@app.get("/state")
async def get_state():
    obs = _current_observation()
    if obs is None:
        return {"observation": None}
    return {"observation": _to_jsonable(obs)}


@app.post("/reset")
async def reset_env(payload: Dict[str, Any] = Body(default_factory=dict)):
    task_name = payload.get("task_name", "single-load")
    obs = singleton_env.reset(task_name=task_name)
    return _coerce_reset_output(obs, task_name)


@app.post("/step")
async def step_env(payload: Dict[str, Any] = Body(default_factory=dict)):
    action_data = payload.get("action")
    if not isinstance(action_data, dict):
        raise HTTPException(status_code=400, detail="Missing action")

    action_payload = _build_action_payload(action_data)

    try:
        action = ModelFlowAction(**action_payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}") from e

    obs = singleton_env.step(action)
    obs_json = _to_jsonable(obs)

    return {
        "observation": obs_json,
        "reward": getattr(obs, "reward", obs_json.get("reward", None) if isinstance(obs_json, dict) else None),
        "done": getattr(obs, "done", obs_json.get("done", None) if isinstance(obs_json, dict) else None),
        "state": {
            "episode_id": "singleton",
            "step_count": getattr(obs, "step_count", obs_json.get("step_count", 0) if isinstance(obs_json, dict) else 0),
        },
    }


@app.get("/schema")
async def get_schema():
    return {
        "action": _schema_for(ModelFlowAction),
        "observation": _schema_for(ModelFlowObservation),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "hello", "message": "ModelFlow websocket connected"})

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "state":
                obs = _current_observation()
                await ws.send_json({"type": "state", "observation": _to_jsonable(obs) if obs is not None else None})

            elif msg_type == "schema":
                await ws.send_json(
                    {
                        "type": "schema",
                        "action": _schema_for(ModelFlowAction),
                        "observation": _schema_for(ModelFlowObservation),
                    }
                )

            elif msg_type == "reset":
                task_name = msg.get("task_name", "single-load")
                obs = singleton_env.reset(task_name=task_name)
                await ws.send_json({"type": "reset", **_coerce_reset_output(obs, task_name)})

            elif msg_type == "step":
                action_data = msg.get("action")
                if not isinstance(action_data, dict):
                    await ws.send_json({"type": "error", "detail": "Missing action"})
                    continue

                action_payload = _build_action_payload(action_data)
                action = ModelFlowAction(**action_payload)
                obs = singleton_env.step(action)
                obs_json = _to_jsonable(obs)

                await ws.send_json(
                    {
                        "type": "step",
                        "observation": obs_json,
                        "reward": getattr(obs, "reward", obs_json.get("reward", None) if isinstance(obs_json, dict) else None),
                        "done": getattr(obs, "done", obs_json.get("done", None) if isinstance(obs_json, dict) else None),
                    }
                )

            else:
                await ws.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        return


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the ModelFlow server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()