# """
# FastAPI application for the Test Environment.

# This module creates an HTTP server that exposes the TestEnvironment
# over HTTP and WebSocket endpoints, compatible with EnvClient.

# Endpoints:
#     - POST /reset: Reset the environment
#     - POST /step: Execute an action
#     - GET /state: Get current environment state
#     - GET /schema: Get action/observation schemas
#     - WS /ws: WebSocket endpoint for persistent sessions

# Usage:
#     # Development (with auto-reload):
#     uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

#     # Production:
#     uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

#     # Or run directly:
#     python -m server.app
# """

# try:
#     from openenv.core.env_server.http_server import create_app
# except Exception as e:  # pragma: no cover
#     raise ImportError(
#         "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
#     ) from e

# import sys
# from pathlib import Path

# # Robust path handling: ensure the environment root is in sys.path
# # This allows 'python app.py' to work even when run from the server/ directory
# _root = Path(__file__).resolve().parent
# if _root.name == "server":
#     _root = _root.parent
# if str(_root) not in sys.path:
#     sys.path.insert(0, str(_root))

# from models import ModelFlowAction, ModelFlowObservation
# from server.test_environment import ModelFlowEnvironment
# from fastapi import Body
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from pathlib import Path


# # Create the app with web interface and README integration
# _readme = _root / "README.md"
# app = create_app(
#     ModelFlowEnvironment,
#     ModelFlowAction,
#     ModelFlowObservation,
#     env_name="modelflow",
#     max_concurrent_envs=1
# )

# # Crucial fix: Override `openenv` default routes to intercept them with a SHARED environment.
# # This allows `inference.py` and the Dashboard to interact with the EXACT SAME state simultaneously.
# new_routes = [
#     r for r in app.router.routes 
#     if getattr(r, "path", None) not in ("/reset", "/step", "/state", "/schema")
# ]
# app.router.routes.clear()
# app.router.routes.extend(new_routes)

# # Robust path for scripts directory
# _scripts_path = _root / "scripts"
# _landing_html = _scripts_path / "landing.html"
# # Mount the static directory for index.jsx
# app.mount("/scripts", StaticFiles(directory=str(_scripts_path)), name="scripts")


 
# # Singleton environment for dashboard Use
# singleton_env = ModelFlowEnvironment()

# @app.get("/")
# async def root():
#     return RedirectResponse(url="/dashboard")

# @app.get("/state")
# async def get_state():
#     """Get the current environment state for the dashboard."""
#     return {"observation": singleton_env._get_observation()}


# @app.post("/reset")
# async def reset_env(payload: dict = Body(...)):
#     """Reset the singleton environment."""
#     task_name = payload.get("task_name", "single-load")
#     obs = singleton_env.reset(task_name=task_name)
#     return {
#         "observation": obs,
#         "state": {"episode_id": "singleton", "step_count": obs.step_count}
#     }


# @app.post("/step")
# async def step_env(payload: dict = Body(...)):
#     """Execute a step in the singleton environment."""
#     action_data = payload.get("action")
#     if not action_data:
#         return {"error": "Missing action"}, 400
    
#     action = ModelFlowAction(**action_data)
#     obs = singleton_env.step(action)
#     return {
#         "observation": obs,
#         "reward": obs.reward,
#         "done": obs.done,
#         "state": {"episode_id": "singleton", "step_count": obs.step_count}
#     }


# @app.get("/schema")
# async def get_schema():
#     """Return JSON schemas for actions and observations."""
#     return {
#         "action": ModelFlowAction.schema(),
#         "observation": ModelFlowObservation.schema()
#     }


# @app.get("/dashboard", response_class=HTMLResponse)
# async def get_dashboard():
#     return """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>ModelFlow Dashboard</title>
#     <script src="https://unpkg.com/react@18/umd/react.development.js" crossorigin></script>
#     <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js" crossorigin></script>
#     <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
#     <link href="https://fonts.googleapis.com/css2?family=VT323&family=Press+Start+2P&display=swap" rel="stylesheet">
#     <style>
#         * { box-sizing: border-box; }
#         body { margin: 0; background: #070502; overflow: hidden; }
#         #root { height: 100vh; width: 100vw; }
#         #boot {
#             position: fixed; inset: 0; background: #070502;
#             display: flex; align-items: center; justify-content: center;
#             flex-direction: column; gap: 16px; z-index: 999;
#             font-family: 'VT323', monospace; color: #00e5ff; font-size: 20px;
#         }
#         #boot-title {
#             font-family: 'Press Start 2P', monospace;
#             font-size: 13px; letter-spacing: 3px; margin-bottom: 8px;
#         }
#         #boot-bar-wrap {
#             width: 260px; height: 6px; background: #1a1a2e; border-radius: 3px; overflow: hidden;
#         }
#         #boot-bar {
#             height: 100%; width: 0%; background: #00e5ff;
#             transition: width 0.3s ease; border-radius: 3px;
#         }
#         #boot-msg { font-size: 16px; color: #334; min-height: 22px; }
#     </style>
# </head>
# <body>
#     <div id="boot">
#         <div id="boot-title">◈ MODELFLOW v2.0</div>
#         <div id="boot-bar-wrap"><div id="boot-bar"></div></div>
#         <div id="boot-msg">Initialising enhanced dashboard...</div>
#     </div>
#     <div id="root"></div>

#     <script>
#         // Fetch the .jsx source, transpile with Babel, then eval.
#         (async function () {
#             const bar = document.getElementById('boot-bar');
#             const msg = document.getElementById('boot-msg');

#             function setProgress(pct, text) {
#                 bar.style.width = pct + '%';
#                 msg.textContent = text;
#             }

#             try {
#                 setProgress(20, 'Fetching dashboard source...');
#                 const resp = await fetch('/scripts/index.jsx');
#                 if (!resp.ok) throw new Error('Failed to fetch /scripts/index.jsx: ' + resp.status);
#                 const src = await resp.text();

#                 setProgress(55, 'Transpiling JSX...');
#                 await new Promise(r => setTimeout(r, 30));

#                 const transformed = Babel.transform(src, {
#                     presets: ['react'],
#                     plugins: [],
#                 }).code;

#                 setProgress(85, 'Mounting React...');
#                 await new Promise(r => setTimeout(r, 30));

#                 // eslint-disable-next-line no-eval
#                 eval(transformed);

#                 setProgress(100, 'Ready.');
#                 await new Promise(r => setTimeout(r, 200));
#                 document.getElementById('boot').style.display = 'none';

#             } catch (err) {
#                 msg.style.color = '#ff4757';
#                 msg.textContent = 'Error: ' + err.message;
#                 bar.style.background = '#ff4757';
#                 bar.style.width = '100%';
#                 console.error('[ModelFlow boot]', err);
#             }
#         })();
#     </script>
# </body>
# </html>
# """
 

# def main():
#     """
#     Main entry point for the environment server.
#     """
#     import uvicorn
#     import argparse

#     parser = argparse.ArgumentParser(description="Run the ModelFlow environment server.")
#     parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
#     parser.add_argument("--port", type=int, default=8000, help="Binding port")
#     args = parser.parse_args()

#     uvicorn.run(app, host=args.host, port=args.port)


# if __name__ == "__main__":
#     main()


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