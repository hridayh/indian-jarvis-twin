from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from app.ai.schemas import BusinessState
from app.prediction.features import FEATURE_VERSION, state_to_feature_dict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite DB file, e.g. ./jarvis_state.db")
    ap.add_argument("--out", required=True, help="Output directory for trained models, e.g. ./models")
    ap.add_argument(
        "--labels",
        default="./data/labels/labels.csv",
        help="CSV with columns: client_id, as_of, plus one or more binary label columns",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument(
        "--tolerance-days",
        type=int,
        default=35,
        help="Max gap allowed between label as_of and snapshot_time when joining (covers monthly horizon)",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out)
    labels_path = Path(args.labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshots
    eng = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
    snaps = pd.read_sql(
        """
        SELECT c.client_id, s.created_at as snapshot_time, s.state_json
        FROM state_snapshots s
        JOIN clients c ON c.id = s.client_fk
        ORDER BY c.client_id, s.created_at
        """,
        eng,
    )
    if snaps.empty:
        raise SystemExit("No state snapshots found. Run ingestion first to populate jarvis_state.db")

    # Parse JSON -> feature dict
    rows = []
    for r in snaps.itertuples(index=False):
        state = BusinessState(**json.loads(r.state_json))
        feats = state_to_feature_dict(state, recent_events=None)
        feats["client_id"] = r.client_id
        feats["snapshot_time"] = pd.to_datetime(r.snapshot_time, utc=True, errors="coerce")
        rows.append(feats)
    feat_df = pd.DataFrame(rows)
    feat_df = feat_df.dropna(subset=["snapshot_time"])

    # Load labels
    if not labels_path.exists():
        raise SystemExit(
            f"Labels file not found: {labels_path}. Create it with columns: client_id, as_of, plus one or more binary label columns"
        )
    labels = pd.read_csv(labels_path)
    required = {"client_id", "as_of"}
    missing = required - set(labels.columns)
    if missing:
        raise SystemExit(f"Labels CSV missing columns: {sorted(missing)}")

    labels["as_of"] = pd.to_datetime(labels["as_of"], utc=True, errors="coerce")
    labels = labels.dropna(subset=["as_of"])

    # Join labels to nearest snapshot at-or-before as_of per client
    feat_df = feat_df.sort_values(["client_id", "snapshot_time"])
    labels = labels.sort_values(["client_id", "as_of"])
    joined = pd.merge_asof(
        labels,
        feat_df,
        by="client_id",
        left_on="as_of",
        right_on="snapshot_time",
        direction="backward",
        tolerance=pd.Timedelta(f"{args.tolerance_days}D"),
    )
    joined = joined.dropna(subset=["snapshot_time"])

    # Label columns: everything except keys
    key_cols = {"client_id", "as_of", "snapshot_time"}
    label_cols = [c for c in joined.columns if c not in key_cols and c not in feat_df.columns]
    # Safer: label cols are those present in labels but not in keys
    label_cols = [c for c in labels.columns if c not in {"client_id", "as_of"}]
    if not label_cols:
        raise SystemExit("No label columns found. Add at least one binary label column to labels.csv")

    drop_cols = ["client_id", "as_of", "snapshot_time", "state_json"] + label_cols
    X_df = joined.drop(columns=[c for c in drop_cols if c in joined.columns], errors="ignore")

    # Feature columns: deterministic order
    feature_columns = sorted([c for c in X_df.columns])
    X = X_df[feature_columns].to_numpy(dtype=float)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    model_files: dict[str, str] = {}
    model_metrics: dict[str, dict] = {}

    for label_col in label_cols:
        y = joined[label_col].astype(int)
        if y.nunique() < 2:
            print(f"Skipping {label_col}: only one class present")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            missing=np.nan,
            eval_metric="logloss",
        )
        model.fit(X_tr, y_tr)

        auc = None
        try:
            auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        except Exception:
            auc = None

        fname = f"{label_col}_xgb.json"
        out_path = out_dir / fname
        model.save_model(str(out_path))
        model_files[label_col] = fname
        model_metrics[label_col] = {"auc": auc, "n_rows": int(len(joined)), "pos_rate": float(y.mean())}

    if not model_files:
        raise SystemExit("No models were trained (check that your label columns contain both 0 and 1).")

    meta = {
        "feature_version": FEATURE_VERSION,
        "feature_columns": feature_columns,
        "models": model_files,
        "metrics": model_metrics,
    }
    with open(out_dir / "risk_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved models:")
    for k, v in model_files.items():
        print(f"- {k}: {out_dir / v}")
    print(f"- metadata: {out_dir / 'risk_model_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

