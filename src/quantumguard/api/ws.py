from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Fan-out of typed JSON events to every connected dashboard client.

    Events are dicts with a "type" key ("health", "predictions", "reasoning",
    "candidate", "deployed", "simulation"). A bounded replay buffer lets a
    client that connects mid-simulation backfill the story so far.
    """

    REPLAY_BUFFER_SIZE = 400

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._replay: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
            replay = list(self._replay)
        for event in replay:
            await websocket.send_text(json.dumps(event))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def broadcast(self, event: dict[str, Any]) -> None:
        async with self._lock:
            self._replay.append(event)
            if len(self._replay) > self.REPLAY_BUFFER_SIZE:
                self._replay = self._replay[-self.REPLAY_BUFFER_SIZE :]
            connections = list(self._connections)
        message = json.dumps(event)
        dead = []
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            await self.disconnect(websocket)

    def reset_replay(self) -> None:
        self._replay = []
