from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import ManualEvalSettings, default_settings
from .schemas import SessionResponse, SubmitActionRequest
from .service import ManualEvalService


def create_app(settings: ManualEvalSettings | None = None) -> FastAPI:
    app_settings = settings or default_settings()
    service = ManualEvalService(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.initialize()
        yield

    app = FastAPI(title="Manual Ragebait Evaluation API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/session", response_model=SessionResponse)
    def get_session() -> dict[str, object]:
        return service.get_session()

    @app.post("/api/respond", response_model=SessionResponse)
    def respond(payload: SubmitActionRequest) -> dict[str, object]:
        try:
            return service.submit_action(payload.action, payload.label)
        except ValueError as exc:
            message = str(exc)
            status_code = 409 if "No tweets remain" in message else 400
            raise HTTPException(status_code=status_code, detail=message) from exc

    @app.post("/api/undo", response_model=SessionResponse)
    def undo() -> dict[str, object]:
        return service.undo()

    return app


app = create_app()
