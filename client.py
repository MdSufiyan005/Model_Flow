# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ModelFlow Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ModelFlowAction, ModelFlowObservation


class ModelFlowEnv(
    EnvClient[ModelFlowAction, ModelFlowObservation, State]
):
    """
    Client for the ModelFlow Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ModelFlowEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.ram_used_mb)
        ...
        ...     result = client.step(ModelFlowAction(command="IDLE"))
        ...     print(result.observation.step_count)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ModelFlowEnv.from_docker_image("modelflow_test:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ModelFlowAction(command="IDLE"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ModelFlowAction) -> Dict:
        """
        Convert ModelFlowAction to JSON payload for step message.
        """
        return action.dict(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ModelFlowObservation]:
        """
        Parse server response into StepResult[ModelFlowObservation].
        """
        obs_data = payload.get("observation", {})
        observation = ModelFlowObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
