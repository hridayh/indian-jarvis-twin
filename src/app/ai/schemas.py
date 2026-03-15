from __future__ import annotations

from pydantic import BaseModel, Field


class InventorySignal(BaseModel):
    sku: str
    qty_estimate: float | None = None
    level_pct: float | None = Field(default=None, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1)


class BusinessState(BaseModel):
    """
    Normalized per-client business state extracted from text + vision.
    Keep this small and stable; store raw evidence separately.
    """

    # Sales / demand
    demand_signal: str | None = None  # e.g. "high", "normal", "low"
    recent_sales_amount_inr: float | None = None

    # Credit / payments
    credit_outstanding_inr: float | None = None
    payment_due_days: int | None = None

    # Stock
    inventory: list[InventorySignal] = Field(default_factory=list)

    # Freeform summary for explainability
    summary: str | None = None


class OrchestrationResult(BaseModel):
    client_id: str
    recommendation_text: str
    risk: dict
    updated_state: BusinessState
    tts_audio_wav_b64: str | None = None

