from __future__ import annotations

import json

from app.ai.schemas import BusinessState, InventorySignal


class OllamaBusinessExtractor:
    """
    Open-source LLM extraction using Ollama.

    Produces a normalized BusinessState from:
    - transcribed Hinglish/Hindi text
    - optional vision inventory signals
    """

    SYSTEM_PROMPT = """You are an expert operations assistant for Indian wholesale businesses.
Extract a compact JSON object with these fields:
- demand_signal: "high"|"normal"|"low"|null
- recent_sales_amount_inr: number|null
- credit_outstanding_inr: number|null
- payment_due_days: integer|null
- inventory: list of {sku, qty_estimate|null, level_pct|null, confidence|null}
- summary: short Hinglish/Hindi summary
Return ONLY valid JSON, no markdown, no extra keys."""

    def __init__(self, *, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def extract(self, *, transcript: str, vision_payload: dict | None = None) -> BusinessState:
        prompt = _build_prompt(transcript=transcript, vision_payload=vision_payload)
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
        data = _safe_json(content)
        return BusinessState(**data)


def _build_prompt(*, transcript: str, vision_payload: dict | None) -> str:
    vp = json.dumps(vision_payload or {}, ensure_ascii=False)
    return f"""Transcript (Hinglish/Hindi):
{transcript}

Vision signals (JSON):
{vp}
"""


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

