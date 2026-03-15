from __future__ import annotations

import io

from app.ai.schemas import InventorySignal


class UltralyticsInventoryVision:
    """
    Vision wrapper for inventory detection from CCTV snapshots.

    - This is scaffolded around ultralytics YOLO models.
    - "YOLOv11" naming varies by release; ultralytics package handles supported weights.
    """

    def __init__(self, *, model_path: str | None = None):
        self.model_path = model_path
        self._model = None

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
        try:
            model = self._lazy_model()
            # ultralytics can accept bytes via PIL; keep minimal dependency by importing PIL lazily
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            _ = model.predict(img, verbose=False)
        except Exception:
            # If model inference fails, still produce a valid empty signal
            pass

        return {
            "inventory": [
                InventorySignal(sku="mixed_inventory", level_pct=None, confidence=None).model_dump()
            ],
            "notes": "vision scaffold; plug in a SKU-specific detector to produce real estimates",
        }

