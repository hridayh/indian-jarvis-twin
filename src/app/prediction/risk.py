from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ai.schemas import BusinessState


@dataclass(frozen=True)
class RiskPrediction:
    stockout_risk: float  # 0..1
    payment_delay_risk: float  # 0..1


class RiskPredictor:
    """
    XGBoost-based risk predictor scaffold.

    In production you would:
    - build a labeled dataset from `events`/`states`
    - train separate models for stockout and payment-delay
    - load those models here and run inference.

    This baseline returns heuristic risks, but keeps the interface stable.
    """

    def __init__(self):
        self.stockout_model = None
        self.payment_model = None

    def predict(self, state: BusinessState, recent_events: list[dict]) -> RiskPrediction:
        # Heuristic baseline (replace with XGB inference).
        inv_levels = [i.level_pct for i in state.inventory if i.level_pct is not None]
        inv_level = float(np.mean(inv_levels)) if inv_levels else 60.0
        stockout_risk = float(np.clip((40.0 - inv_level) / 40.0, 0.0, 1.0))

        due = state.payment_due_days or 0
        credit = state.credit_outstanding_inr or 0.0
        payment_delay_risk = float(np.clip((due / 30.0) * (1.0 if credit > 0 else 0.5), 0.0, 1.0))
        return RiskPrediction(stockout_risk=stockout_risk, payment_delay_risk=payment_delay_risk)

