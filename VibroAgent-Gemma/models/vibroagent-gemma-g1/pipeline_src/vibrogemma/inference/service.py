from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .store import EventStore


def create_app(store: EventStore, dashboard_dir: str | Path) -> FastAPI:
    dashboard = Path(dashboard_dir)
    app = FastAPI(title="VibroGemma runtime", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "latest_event_id": (store.latest() or {}).get("id")}

    @app.get("/latest")
    def latest() -> dict[str, Any] | None:
        return store.latest()

    @app.get("/alerts")
    def alerts(
        limit: int = Query(100, ge=1, le=10_000),
        alerts_only: bool = Query(False),
    ) -> list[dict[str, Any]]:
        return store.list(limit=limit, alerts_only=alerts_only)

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        last_id = None
        try:
            while True:
                latest_event = store.latest()
                event_id = latest_event.get("id") if latest_event else None
                if event_id != last_id:
                    await websocket.send_json(latest_event or {"status": "waiting"})
                    last_id = event_id
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    if dashboard.exists():
        app.mount("/static", StaticFiles(directory=dashboard), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(dashboard / "index.html")

    return app
