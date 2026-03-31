from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes_demo import router as demo_router
from app.api.routes_ingest import router as ingest_router
from app.config import Settings
from app.digital_twin.sqlite_store import SQLiteStateStore
from app.orchestrator.twin_orchestrator import TwinOrchestrator
from app.ai.llm import OllamaBusinessExtractor
from app.ai.schemas import BusinessState, InventorySignal
from app.ai.stt import WhisperSTT
from app.ai.vision import UltralyticsInventoryVision
from app.ai.tts import build_tts
from app.prediction.risk import RiskPredictor


_DEMO_CLIENT = "+91-9876543210"

def _seed_demo_state(state_store) -> None:
    """If the demo client has no stored state, insert a realistic baseline."""
    import logging
    log = logging.getLogger(__name__)
    existing = state_store.get_latest_state(_DEMO_CLIENT)
    # Only skip if there's inventory with actual level data (not garbage null items)
    if existing and any(i.level_pct is not None for i in existing.inventory):
        return
    log.info("[Startup] Seeding initial Digital Twin state for demo client %s", _DEMO_CLIENT)
    state_store.ensure_client(_DEMO_CLIENT)
    seed = BusinessState(
        demand_signal="normal",
        recent_sales_amount_inr=75_000.0,
        credit_outstanding_inr=42_000.0,
        payment_due_days=12,
        inventory=[
            InventorySignal(sku="SKU_COTTON_001", level_pct=62.0, qty_estimate=12.4, confidence=0.8),
            InventorySignal(sku="SKU_POLY_002",   level_pct=47.0, qty_estimate=9.4,  confidence=0.8),
            InventorySignal(sku="SKU_SILK_003",   level_pct=78.0, qty_estimate=15.6, confidence=0.8),
            InventorySignal(sku="SKU_SHIRTBOX_004", level_pct=31.0, qty_estimate=6.2, confidence=0.75),
            InventorySignal(sku="SKU_SAREE_005",  level_pct=55.0, qty_estimate=11.0, confidence=0.8),
        ],
        summary="Baseline state seeded at server startup for demo.",
    )
    state_store.set_latest_state(_DEMO_CLIENT, seed)
    log.info("[Startup] ✅ Demo state seeded — %d SKUs", len(seed.inventory))


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="Voice AI Digital Twin — SMB Operator Assistant",
        version="0.1.0",
    )

    # Allow the frontend (any origin in dev) to call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten to specific origin in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core services (DI via app.state)
    state_store = SQLiteStateStore(db_url=settings.state_store_sqlite_url)
    stt = WhisperSTT(model_name=settings.stt_model_name, device=settings.stt_device)
    vision = UltralyticsInventoryVision(
        model_path=settings.vision_model_path,
        sku_mapping_path=settings.vision_sku_mapping_path,
    )
    extractor = OllamaBusinessExtractor(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    tts = build_tts(provider=settings.tts_provider, model_name=settings.tts_model_name)
    risk_predictor = RiskPredictor(
        model_dir=settings.risk_model_dir,
        stockout_label=settings.risk_stockout_label,
        payment_delay_label=settings.risk_payment_delay_label,
    )

    app.state.settings = settings
    app.state.state_store = state_store
    app.state.orchestrator = TwinOrchestrator(
        state_store=state_store,
        stt=stt,
        vision=vision,
        extractor=extractor,
        tts=tts,
        risk_predictor=risk_predictor,
    )

    # Seed demo client with realistic initial inventory if they have no state yet.
    # This means even on a fresh DB, "aaj ka update do" returns meaningful data.
    _seed_demo_state(state_store)

    app.include_router(ingest_router, tags=["ingestion"])
    app.include_router(demo_router)

    # Serve the frontend SPA at /ui  (open http://localhost:8000/ui/index.html)
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


app = create_app()
