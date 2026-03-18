from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ai.schemas import BusinessState
from app.prediction.features import FEATURE_VERSION, state_to_feature_dict, vectorize


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

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        stockout_label: str = "stockout_lowstock_w1",
        payment_delay_label: str = "payment_overdue14_w1",
    ):
        self.model_dir = model_dir
        self.stockout_label = stockout_label
        self.payment_delay_label = payment_delay_label
        self.stockout_model = None
        self.payment_model = None
        self.feature_columns: list[str] | None = None

        if self.model_dir:
            self._try_load_models()

    def _try_load_models(self) -> None:
        try:
            import json
            import os

            import xgboost as xgb
        except Exception:
            return

        meta_path = os.path.join(self.model_dir, "risk_model_metadata.json")
        if not os.path.exists(meta_path):
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("feature_version") != FEATURE_VERSION:
            return

        self.feature_columns = list(meta["feature_columns"])

        # New format: models mapping label_col -> filename
        models = meta.get("models")
        if isinstance(models, dict):
            stock_file = models.get(self.stockout_label)
            pay_file = models.get(self.payment_delay_label)
            if not (stock_file and pay_file):
                return
            stock_path = os.path.join(self.model_dir, stock_file)
            pay_path = os.path.join(self.model_dir, pay_file)
            if not (os.path.exists(stock_path) and os.path.exists(pay_path)):
                return
        else:
            # Backward-compatible old format filenames
            stock_path = os.path.join(self.model_dir, "stockout_xgb.json")
            pay_path = os.path.join(self.model_dir, "payment_delay_xgb.json")
            if not (os.path.exists(stock_path) and os.path.exists(pay_path)):
                return

        self.stockout_model = xgb.XGBClassifier()
        self.stockout_model.load_model(stock_path)

        self.payment_model = xgb.XGBClassifier()
        self.payment_model.load_model(pay_path)

    def predict(self, state: BusinessState, recent_events: list[dict]) -> RiskPrediction:
        # If trained models are present, use them; else fallback to heuristics.
        if self.stockout_model and self.payment_model and self.feature_columns:
            feats = state_to_feature_dict(state, recent_events=recent_events)
            X = vectorize(feats, self.feature_columns)
            try:
                stockout_risk = float(self.stockout_model.predict_proba(X)[0, 1])
                payment_delay_risk = float(self.payment_model.predict_proba(X)[0, 1])
                return RiskPrediction(stockout_risk=stockout_risk, payment_delay_risk=payment_delay_risk)
            except Exception:
                # model mismatch or runtime error; fall through to heuristics
                pass

        # Heuristic baseline.
        inv_levels = [i.level_pct for i in state.inventory if i.level_pct is not None]
        inv_level = float(np.mean(inv_levels)) if inv_levels else 60.0
        stockout_risk = float(np.clip((40.0 - inv_level) / 40.0, 0.0, 1.0))

        due = state.payment_due_days or 0
        credit = state.credit_outstanding_inr or 0.0
        payment_delay_risk = float(np.clip((due / 30.0) * (1.0 if credit > 0 else 0.5), 0.0, 1.0))
        return RiskPrediction(stockout_risk=stockout_risk, payment_delay_risk=payment_delay_risk)

