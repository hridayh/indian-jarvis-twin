from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_ingest import router as ingest_router
from app.config import Settings
from app.digital_twin.sqlite_store import SQLiteStateStore
from app.orchestrator.twin_orchestrator import TwinOrchestrator
from app.ai.llm import OllamaBusinessExtractor
from app.ai.stt import WhisperSTT
from app.ai.vision import UltralyticsInventoryVision
from app.ai.tts import CoquiTTS


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="Indian Jarvis — Digital Twin Backend",
        version="0.1.0",
    )

    # Core services (DI via app.state)
    state_store = SQLiteStateStore(db_url=settings.state_store_sqlite_url)
    stt = WhisperSTT(model_name=settings.stt_model_name, device=settings.stt_device)
    vision = UltralyticsInventoryVision(model_path=settings.vision_model_path)
    extractor = OllamaBusinessExtractor(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    tts = CoquiTTS(model_name=settings.tts_model_name)

    app.state.settings = settings
    app.state.state_store = state_store
    app.state.orchestrator = TwinOrchestrator(
        state_store=state_store,
        stt=stt,
        vision=vision,
        extractor=extractor,
        tts=tts,
    )

    app.include_router(ingest_router, tags=["ingestion"])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


app = create_app()

