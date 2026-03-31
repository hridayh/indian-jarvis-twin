from __future__ import annotations

import json
import logging

from app.ai.schemas import BusinessState, InventorySignal

logger = logging.getLogger(__name__)


class OllamaBusinessExtractor:
    """
    Open-source LLM extraction using Ollama.

    Produces a normalized BusinessState from:
    - transcribed Hinglish/Hindi text
    - optional vision inventory signals
    """

    SYSTEM_PROMPT = """You are an expert operations assistant for Indian wholesale businesses.
Extract a compact JSON object with EXACTLY these fields:
- demand_signal: one of "high", "normal", "low", or null
- recent_sales_amount_inr: a number (float) or null
- credit_outstanding_inr: a number (float) or null
- payment_due_days: an integer or null
- inventory: list of objects, each with:
    - sku: string
    - qty_estimate: a number (float) or null  — NEVER a string
    - level_pct: a number between 0 and 100 (float) or null  — NEVER a string like "critical"
    - confidence: a number between 0 and 1 (float) or null
- summary: short Hinglish/Hindi plain-text summary (string)
RULES: Return ONLY valid JSON. No markdown. No extra keys. All numeric fields must be numbers or null."""

    def __init__(self, *, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def extract(self, *, transcript: str, vision_payload: dict | None = None) -> BusinessState:
        prompt = _build_prompt(transcript=transcript, vision_payload=vision_payload)
        logger.info("[LLM] Sending to %s @ %s", self.model, self.base_url)
        logger.info("[LLM] ── Prompt ──────────────────────────────────────────")
        logger.info("[LLM] Transcript: %r", transcript[:300])
        logger.info("[LLM] Vision:     %s", json.dumps(vision_payload or {}, ensure_ascii=False)[:200])

        try:
            import ollama
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ollama python client not installed/working. Install deps or swap LLM implementation."
            ) from e

        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["message"]["content"]
        logger.info("[LLM] ── Raw response ─────────────────────────────────────")
        logger.info("[LLM] %s", content[:800])

        try:
            data = _safe_json(content)
        except ValueError as e:
            logger.error("[LLM] ❌ JSON parse failed: %s — raw: %r", e, content[:300])
            raise

        before = json.dumps(data, ensure_ascii=False)
        data = _sanitize(data)
        after = json.dumps(data, ensure_ascii=False)
        if before != after:
            logger.warning("[LLM] Sanitizer changed output:")
            logger.warning("[LLM]   before: %s", before[:300])
            logger.warning("[LLM]   after:  %s", after[:300])

        logger.info("[LLM] ✅ Parsed → demand=%s sales=%s credit=%s inv=%d items",
                    data.get("demand_signal"), data.get("recent_sales_amount_inr"),
                    data.get("credit_outstanding_inr"), len(data.get("inventory") or []))
        return BusinessState(**data)


def _build_prompt(*, transcript: str, vision_payload: dict | None) -> str:
    vp = json.dumps(vision_payload or {}, ensure_ascii=False)
    return f"""Transcript (Hinglish/Hindi):
{transcript}

Vision signals (JSON):
{vp}
"""


def _sanitize(data: dict) -> dict:
    """
    Coerce LLM output into types Pydantic expects.
    LLMs sometimes return strings like 'critical' or 'low' for numeric fields.
    """
    # Numeric top-level fields
    for key in ("recent_sales_amount_inr", "credit_outstanding_inr", "payment_due_days"):
        val = data.get(key)
        if val is not None:
            try:
                data[key] = float(val) if key != "payment_due_days" else int(float(val))
            except (TypeError, ValueError):
                data[key] = None

    # Inventory items: strip non-numeric values from numeric fields
    inv = data.get("inventory")
    if isinstance(inv, list):
        clean = []
        for item in inv:
            if not isinstance(item, dict):
                continue
            for field in ("level_pct", "qty_estimate", "confidence"):
                v = item.get(field)
                if v is not None:
                    try:
                        item[field] = float(v)
                    except (TypeError, ValueError):
                        item[field] = None  # drop bad strings silently
            clean.append(item)
        data["inventory"] = clean

    return data


def _safe_json(text: str) -> dict:
    """
    Handles minor LLM formatting issues (leading/trailing text).
    """
    text = text.strip()
    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("LLM did not return valid JSON")

