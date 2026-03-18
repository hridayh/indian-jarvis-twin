from __future__ import annotations

import io
import json
from pathlib import Path

from app.ai.schemas import InventorySignal


class UltralyticsInventoryVision:
    """
    Vision wrapper for inventory detection from CCTV snapshots.

    - This is scaffolded around ultralytics YOLO models.
    - "YOLOv11" naming varies by release; ultralytics package handles supported weights.
    """

    def __init__(self, *, model_path: str | None = None, sku_mapping_path: str | None = None):
        self.model_path = model_path
        self.sku_mapping_path = sku_mapping_path
        self._model = None
        self._mapping = None

    def _load_mapping(self) -> dict:
        if self._mapping is not None:
            return self._mapping
        if not self.sku_mapping_path:
            self._mapping = {}
            return self._mapping
        p = Path(self.sku_mapping_path)
        if not p.exists():
            self._mapping = {}
            return self._mapping
        self._mapping = json.loads(p.read_text(encoding="utf-8"))
        return self._mapping

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ultralytics not installed/working. Install deps or swap vision implementation."
            ) from e

        # If model_path not provided, fall back to a tiny pretrained detector
        weights = self.model_path or "yolov8n.pt"
        self._model = YOLO(weights)
        return self._model

    def detect_inventory(self, image_bytes: bytes) -> dict:
        """
        Returns dict with high-level signals + per-SKU estimates.

        NOTE: Real inventory estimation requires a domain-specific model + calibration:
        camera geometry, SKU shelf mapping, etc. Here we return a stub-friendly structure.
        """
        mapping = self._load_mapping()
        try:
            model = self._lazy_model()
            # ultralytics can accept bytes via PIL; keep minimal dependency by importing PIL lazily
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = model.predict(img, verbose=False)

            # Convert detections -> class counts (very rough proxy)
            class_counts: dict[str, int] = {}
            if results:
                r0 = results[0]
                names = getattr(r0, "names", {}) or {}
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and getattr(boxes, "cls", None) is not None:
                    for c in boxes.cls.tolist():
                        cname = names.get(int(c), str(int(c)))
                        class_counts[cname] = class_counts.get(cname, 0) + 1

            inv = _map_classes_to_inventory(class_counts, mapping)
            if inv:
                return {"inventory": [i.model_dump() for i in inv], "notes": "mapped from YOLO class counts (MVP proxy)"}
        except Exception:
            # If model inference fails, still produce a valid empty signal
            pass

        return {
            "inventory": [
                InventorySignal(sku="mixed_inventory", level_pct=None, confidence=None).model_dump()
            ],
            "notes": "vision scaffold; plug in a SKU-specific detector to produce real estimates",
        }


def _map_classes_to_inventory(class_counts: dict[str, int], mapping: dict) -> list[InventorySignal]:
    """
    MVP mapping mode: YOLO class -> SKU, and class count -> level_pct via a simple scaling.

    Mapping JSON (synthetic MVP):
    {
      "mode": "class_to_sku",
      "class_to_sku": {"cotton_roll": "SKU_COTTON_001"},
      "count_to_level_pct": {"min_count": 0, "max_count": 20}
    }
    """
    if not class_counts:
        return []
    if (mapping or {}).get("mode") != "class_to_sku":
        return []
    c2s = (mapping or {}).get("class_to_sku") or {}
    c2s = {str(k): str(v) for k, v in c2s.items()}
    scale = (mapping or {}).get("count_to_level_pct") or {}
    min_c = float(scale.get("min_count", 0))
    max_c = float(scale.get("max_count", 20))
    if max_c <= min_c:
        max_c = min_c + 1.0

    inv: list[InventorySignal] = []
    for cname, cnt in class_counts.items():
        sku = c2s.get(cname)
        if not sku:
            continue
        level = (float(cnt) - min_c) / (max_c - min_c) * 100.0
        level = max(0.0, min(100.0, level))
        inv.append(InventorySignal(sku=sku, qty_estimate=float(cnt), level_pct=level, confidence=0.5))
    return inv

