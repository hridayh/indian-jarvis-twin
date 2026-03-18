from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from app.ai.schemas import BusinessState


FEATURE_VERSION = 1


def state_to_feature_dict(state: BusinessState, recent_events: list[dict] | None = None) -> dict[str, float]:
    """
    Convert the current BusinessState (+ recent events) into a flat numeric feature dict.

    Keep this deterministic and stable; training and inference both use this.
    """
    recent_events = recent_events or []

    inv_levels = [i.level_pct for i in state.inventory if i.level_pct is not None]
    inv_avg = float(np.mean(inv_levels)) if inv_levels else np.nan
    inv_min = float(np.min(inv_levels)) if inv_levels else np.nan

    demand_map = {"low": 0.0, "normal": 1.0, "high": 2.0}
    demand = demand_map.get((state.demand_signal or "").strip().lower(), np.nan)

    # Event counts (from already-fetched recent events)
    counts: dict[str, float] = {}
    for e in recent_events:
        et = (e.get("event_type") or "").strip()
        if not et:
            continue
        counts[f"evt_cnt_{et}"] = counts.get(f"evt_cnt_{et}", 0.0) + 1.0

    feats: dict[str, float] = {
        "inv_avg_level_pct": inv_avg,
        "inv_min_level_pct": inv_min,
        "inv_known_levels_count": float(len(inv_levels)),
        "credit_outstanding_inr": float(state.credit_outstanding_inr) if state.credit_outstanding_inr is not None else np.nan,
        "payment_due_days": float(state.payment_due_days) if state.payment_due_days is not None else np.nan,
        "recent_sales_amount_inr": float(state.recent_sales_amount_inr) if state.recent_sales_amount_inr is not None else np.nan,
        "demand_signal_ordinal": float(demand) if demand is not np.nan else np.nan,
    }
    feats.update(counts)
    return feats


def vectorize(feature_dict: dict[str, float], feature_columns: list[str]) -> np.ndarray:
    """
    Produce a 2D numpy array shape (1, n_features) in the exact column order.
    Missing features are filled with NaN (XGBoost handles missing if configured).
    """
    row = []
    for c in feature_columns:
        v = feature_dict.get(c, np.nan)
        row.append(v)
    return np.array([row], dtype=float)

