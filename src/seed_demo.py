"""
seed_demo.py  —  Populate jarvis_state.db with synthetic demo data.

Run ONCE before `train.py` or before a demo session that needs history:

    python src/seed_demo.py --db ./jarvis_state.db

What it creates
───────────────
• 3 demo clients (textile wholesale shop owners)
• ~90 state snapshots spread over 90 days per client (realistic drift)
• ~150 raw events per client (whatsapp_voice_note / whatsapp_text / cctv_snapshot)
• data/labels/labels.csv refreshed with correct 0/1 labels matching the snapshots
  (so train.py can be run immediately after)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make sure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sqlalchemy import create_engine, text

from app.ai.schemas import BusinessState, InventorySignal
from app.digital_twin.models import Base, Client, Event, StateSnapshot

# ── Demo config ────────────────────────────────────────────────────────────────

CLIENTS = [
    {"id": "+91-9876543210", "name": "Ravi Textiles, Surat"},
    {"id": "+91-8765432109", "name": "Mehta Fabrics, Ahmedabad"},
    {"id": "+91-7654321098", "name": "Singh Cotton House, Ludhiana"},
]

SKUS = [
    "SKU_COTTON_001",
    "SKU_POLY_002",
    "SKU_SILK_003",
    "SKU_SHIRTBOX_004",
    "SKU_SAREE_005",
]

LOW_STOCK_THRESHOLD = 20.0   # %

SAMPLE_TRANSCRIPTS = [
    "Bhai cotton roll ka stock bahut kam ho gaya, sirf 2-3 bundles bache hain",
    "Mehta ji ka payment abhi tak nahi aaya, 50 hazaar outstanding hai, 18 din ho gaye",
    "Aaj achha din raha, 3 bade orders mile, roughly 80k ka maal gaya",
    "Silk roll ki demand zyada hai is hafte, reorder karna padega jaldi",
    "Polyester wala stock theek hai abhi, 60-70% bhar ke shelf hai",
    "Kal CCTV pe dekha shelves half empty lag rahi thi cotton side mein",
    "Is mahine payment collection slow hai, 2 parties ne partial diya",
    "Stock thoda low ho raha hai, order dena chahiye 2-3 din mein",
    "Aaj saree bundle ke 5 sets nikle, good movement",
    "Credit outstanding bada ho raha hai, 1.2 lakh ho gaya total",
]

EVENT_TYPES = ["whatsapp_voice_note", "whatsapp_text", "cctv_snapshot"]


def _random_state(rng: random.Random, day: int, client_idx: int) -> BusinessState:
    """Generate a realistic BusinessState for a given day index (0-90)."""
    # Inventory drifts down over time and occasionally gets restocked
    base_levels = {
        "SKU_COTTON_001": 70 - day * 0.5 + rng.uniform(-10, 10),
        "SKU_POLY_002":   55 - day * 0.3 + rng.uniform(-8, 8),
        "SKU_SILK_003":   80 - day * 0.4 + rng.uniform(-12, 12),
        "SKU_SHIRTBOX_004": 65 - day * 0.35 + rng.uniform(-10, 10),
        "SKU_SAREE_005":  50 - day * 0.2 + rng.uniform(-8, 8),
    }
    # Simulate restocks at day 30 and 60
    if day > 30:
        base_levels["SKU_COTTON_001"] += 40
        base_levels["SKU_POLY_002"]   += 30
    if day > 60:
        base_levels["SKU_SILK_003"]     += 35
        base_levels["SKU_SHIRTBOX_004"] += 30

    inventory = [
        InventorySignal(
            sku=sku,
            qty_estimate=round(max(0, lvl) / 5, 1),
            level_pct=round(max(0.0, min(100.0, lvl)), 1),
            confidence=0.75,
        )
        for sku, lvl in base_levels.items()
    ]

    # Sales trend
    sales = rng.uniform(20_000, 120_000) * (1 + 0.005 * day)

    # Credit builds up, partially cleared every ~15 days
    credit = rng.uniform(0, 150_000)
    if day % 15 < 3:
        credit *= 0.3   # partial clearance
    due_days = rng.randint(0, 25)

    demand = rng.choices(["low", "normal", "high"], weights=[0.15, 0.6, 0.25])[0]

    return BusinessState(
        demand_signal=demand,
        recent_sales_amount_inr=round(sales, 2),
        credit_outstanding_inr=round(credit, 2),
        payment_due_days=due_days,
        inventory=inventory,
        summary=f"Day {day} snapshot for client {client_idx}",
    )


def _make_label_row(
    client_id: str,
    as_of: datetime,
    state: BusinessState,
    next_state: BusinessState | None,
) -> dict:
    """Derive binary labels by looking at what actually happened 7 and 30 days later."""
    inv_levels = {i.sku: i.level_pct for i in state.inventory if i.level_pct is not None}
    min_lvl = min(inv_levels.values(), default=100.0)

    # For simplicity in synthetic data: derive labels from current + next state
    future_min = min_lvl
    if next_state:
        future_lvls = [i.level_pct for i in next_state.inventory if i.level_pct is not None]
        future_min = min(future_lvls, default=min_lvl)

    stockout_lowstock = int(future_min < LOW_STOCK_THRESHOLD)
    # No-sales proxy: if demand was "low" AND stock is low
    stockout_nosales = int(stockout_lowstock and (state.demand_signal == "low"))

    credit = state.credit_outstanding_inr or 0.0
    due = state.payment_due_days or 0
    payment_overdue14 = int(due > 14 and credit > 0)
    # Partial payment proxy: credit was reduced slightly but not fully
    payment_partial = int(0 < credit < 80_000 and due > 7)

    return {
        "client_id":           client_id,
        "as_of":               as_of.isoformat(),
        "stockout_lowstock_w1": stockout_lowstock,
        "stockout_lowstock_m1": stockout_lowstock,
        "stockout_nosales_w1":  stockout_nosales,
        "stockout_nosales_m1":  stockout_nosales,
        "payment_overdue14_w1": payment_overdue14,
        "payment_overdue14_m1": payment_overdue14,
        "payment_partial_w1":   payment_partial,
        "payment_partial_m1":   payment_partial,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed demo data into jarvis_state.db")
    ap.add_argument("--db",   default="./jarvis_state.db", help="Path to SQLite DB file")
    ap.add_argument("--seed", type=int, default=42,        help="Random seed")
    ap.add_argument("--days", type=int, default=90,        help="Days of history to generate")
    ap.add_argument(
        "--labels-out",
        default="./data/labels/labels.csv",
        help="Where to write the labels CSV",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=args.days)

    db_path = Path(args.db)
    labels_path = Path(args.labels_out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
    Base.metadata.create_all(engine)

    label_rows: list[dict] = []

    with engine.begin() as conn:
        for ci, cinfo in enumerate(CLIENTS):
            client_id = cinfo["id"]
            # Upsert client (must supply created_at — ORM default doesn't fire on raw SQL)
            conn.execute(
                text(
                    "INSERT INTO clients (client_id, created_at) VALUES (:cid, :ts) "
                    "ON CONFLICT(client_id) DO NOTHING"
                ),
                {"cid": client_id, "ts": now.isoformat()},
            )
            row = conn.execute(
                text("SELECT id FROM clients WHERE client_id = :cid"),
                {"cid": client_id},
            ).fetchone()
            fk = row[0]

            states_by_day: dict[int, BusinessState] = {}
            for day in range(args.days + 1):
                ts = start + timedelta(days=day)
                state = _random_state(rng, day, ci)
                states_by_day[day] = state

                # Insert state snapshot
                conn.execute(
                    text(
                        "INSERT INTO state_snapshots (client_fk, state_json, created_at) "
                        "VALUES (:fk, :sj, :ts)"
                    ),
                    {
                        "fk": fk,
                        "sj": state.model_dump_json(),
                        "ts": ts.isoformat(),
                    },
                )

                # Insert 1-2 random events per day
                for _ in range(rng.randint(1, 2)):
                    etype = rng.choice(EVENT_TYPES)
                    transcript = rng.choice(SAMPLE_TRANSCRIPTS)
                    payload: dict = {}
                    if etype in ("whatsapp_voice_note", "whatsapp_text"):
                        payload = {"transcript": transcript}
                    else:
                        payload = {
                            "camera_id": "cam-01",
                            "vision": {"notes": "synthetic cctv event"},
                        }
                    conn.execute(
                        text(
                            "INSERT INTO events (client_fk, event_type, payload_json, created_at) "
                            "VALUES (:fk, :et, :pj, :ts)"
                        ),
                        {
                            "fk":  fk,
                            "et":  etype,
                            "pj":  json.dumps(payload, ensure_ascii=False),
                            "ts":  (ts + timedelta(hours=rng.randint(8, 18))).isoformat(),
                        },
                    )

                # Build label row every 7 days
                if day % 7 == 0 and day > 0:
                    next_state = states_by_day.get(min(day + 7, args.days))
                    label_rows.append(
                        _make_label_row(client_id, ts, state, next_state)
                    )

    # Write labels CSV
    df = pd.DataFrame(label_rows)
    df.to_csv(labels_path, index=False)

    print(f"\n✅  Seeded {args.days} days × {len(CLIENTS)} clients")
    print(f"   DB    → {db_path.resolve()}")
    print(f"   Labels → {labels_path.resolve()}")
    print(f"   Label rows: {len(df)}")
    print(f"\nNext: python src/train.py --db {args.db} --out ./models\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
