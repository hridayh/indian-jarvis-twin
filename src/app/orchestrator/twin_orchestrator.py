from __future__ import annotations

import logging
from dataclasses import dataclass

from app.ai.schemas import BusinessState, OrchestrationResult

logger = logging.getLogger(__name__)
from app.digital_twin.state_store import StateStore
from app.ingestion.twilio_whatsapp import TwilioWhatsAppWebhook
from app.prediction.risk import RiskPredictor
from app.utils.http import download_bytes


@dataclass
class TwinOrchestrator:
    state_store: StateStore
    stt: object
    vision: object
    extractor: object
    tts: object
    risk_predictor: RiskPredictor | None = None

    def __post_init__(self):
        if self.risk_predictor is None:
            self.risk_predictor = RiskPredictor()

    async def process_whatsapp_webhook(self, webhook: TwilioWhatsAppWebhook) -> OrchestrationResult:
        client = webhook.client_id
        logger.info("[Orchestrator] ── New request for client=%s ──────────────────", client)
        self.state_store.ensure_client(client)

        if webhook.has_audio:
            # Use injected bytes (demo route) or fetch from Twilio URL.
            if webhook._audio_bytes_override is not None:
                audio_bytes = webhook._audio_bytes_override
            else:
                # NOTE: Twilio media URLs often require auth. This scaffold does
                # unauthenticated fetch; wire Basic Auth (AccountSid/AuthToken) if needed.
                audio_bytes = download_bytes(webhook.media_url_0)
            logger.info("[Orchestrator] STEP 1/5 — STT: transcribing %d bytes of audio…", len(audio_bytes))
            stt_out = self.stt.transcribe(audio_bytes, content_type=webhook.media_content_type_0)
            transcript = stt_out["text"]
            self.state_store.append_event(
                client,
                event_type="whatsapp_voice_note",
                payload={"transcript": transcript, "stt": stt_out},
            )
        else:
            transcript = (webhook.body or "").strip()
            logger.info("[Orchestrator] STEP 1/5 — Text input (no STT): %r", transcript[:120])
            self.state_store.append_event(
                client,
                event_type="whatsapp_text",
                payload={"text": transcript},
            )

        logger.info("[Orchestrator] STEP 2/5 — LLM: extracting BusinessState from transcript…")
        vision_payload = None
        state = self.extractor.extract(transcript=transcript, vision_payload=vision_payload)
        logger.info("[Orchestrator] LLM result: demand=%s sales=₹%s credit=₹%s inv=%d items",
                    state.demand_signal, state.recent_sales_amount_inr,
                    state.credit_outstanding_inr, len(state.inventory))

        logger.info("[Orchestrator] STEP 3/5 — Digital Twin: merging + persisting state…")
        merged = _merge_state(self.state_store.get_latest_state(client), state)
        self.state_store.set_latest_state(client, merged)

        logger.info("[Orchestrator] STEP 4/5 — XGBoost: scoring risk…")
        recent_events = self.state_store.get_recent_events(client, limit=200)
        risk = self.risk_predictor.predict(merged, recent_events)
        recommendation = _recommendation_text(merged, risk)
        tts_text = _recommendation_text._last_tts
        logger.info("[Orchestrator] Risk: stockout=%.0f%% payment_delay=%.0f%%",
                    risk.stockout_risk * 100, risk.payment_delay_risk * 100)
        logger.info("[Orchestrator] UI text:  %r", recommendation)
        logger.info("[Orchestrator] TTS text: %r", tts_text)

        logger.info("[Orchestrator] STEP 5/5 — TTS: synthesizing Hindi voice response…")
        tts_b64 = None
        try:
            wav = self.tts.synthesize_wav(tts_text, language="hi")
            tts_b64 = self.tts.wav_bytes_to_b64(wav)
            logger.info("[Orchestrator] ✅ TTS done — audio %d bytes → b64 len=%d",
                        len(wav), len(tts_b64))
        except Exception as e:
            # TTS is optional; don't fail ingestion — text response still goes out.
            import traceback
            tb_lines = traceback.format_exc().splitlines()
            logger.warning("[Orchestrator] ⚠ TTS FAILED (text response still sent to frontend):")
            logger.warning("[Orchestrator]   %s: %s", type(e).__name__, e)
            if len(tb_lines) >= 2:
                logger.warning("[Orchestrator]   %s", tb_lines[-2])

        logger.info("[Orchestrator] ── Request complete for client=%s ──────────────", client)
        return OrchestrationResult(
            client_id=client,
            transcript=transcript,
            recommendation_text=recommendation,
            risk={"stockout": risk.stockout_risk, "payment_delay": risk.payment_delay_risk},
            updated_state=merged,
            llm_extracted=state,
            tts_audio_wav_b64=tts_b64,
        )

    async def process_cctv_snapshot(
        self,
        *,
        client_id: str,
        camera_id: str | None,
        image_bytes: bytes,
        content_type: str | None,
    ) -> OrchestrationResult:
        self.state_store.ensure_client(client_id)

        vision_payload = self.vision.detect_inventory(image_bytes)
        self.state_store.append_event(
            client_id,
            event_type="cctv_snapshot",
            payload={"camera_id": camera_id, "content_type": content_type, "vision": vision_payload},
        )

        # No transcript for this event; we can still ask LLM to summarize vision if needed.
        # For now we merge inventory into state directly.
        state = BusinessState(
            inventory=[
                *[
                    _inv_from_dict(d)
                    for d in (vision_payload.get("inventory") or [])
                    if isinstance(d, dict)
                ]
            ],
            summary=(vision_payload.get("notes") if isinstance(vision_payload, dict) else None),
        )
        merged = _merge_state(self.state_store.get_latest_state(client_id), state)
        self.state_store.set_latest_state(client_id, merged)

        recent_events = self.state_store.get_recent_events(client_id, limit=200)
        risk = self.risk_predictor.predict(merged, recent_events)
        recommendation = _recommendation_text(merged, risk)
        tts_text = _recommendation_text._last_tts

        tts_b64 = None
        try:
            wav = self.tts.synthesize_wav(tts_text, language="hi")
            tts_b64 = self.tts.wav_bytes_to_b64(wav)
            logger.info("[Orchestrator/CCTV] ✅ TTS done — %d bytes", len(wav))
        except Exception as e:
            import traceback
            tb_lines = traceback.format_exc().splitlines()
            logger.warning("[Orchestrator/CCTV] ⚠ TTS FAILED: %s: %s", type(e).__name__, e)
            if len(tb_lines) >= 2:
                logger.warning("[Orchestrator/CCTV]   %s", tb_lines[-2])

        return OrchestrationResult(
            client_id=client_id,
            recommendation_text=recommendation,
            risk={"stockout": risk.stockout_risk, "payment_delay": risk.payment_delay_risk},
            updated_state=merged,
            tts_audio_wav_b64=tts_b64,
        )


def _merge_state(prev: BusinessState | None, new: BusinessState) -> BusinessState:
    """
    Simple merge:
    - fill missing numeric fields from previous snapshot
    - prefer new inventory if provided; else keep previous
    """
    if prev is None:
        return new

    merged = prev.model_copy(deep=True)

    # Scalars
    for field in [
        "demand_signal",
        "recent_sales_amount_inr",
        "credit_outstanding_inr",
        "payment_due_days",
        "summary",
    ]:
        val = getattr(new, field)
        if val is not None:
            setattr(merged, field, val)

    # Inventory — merge by SKU: keep old level_pct/qty if new item doesn't have them
    # First strip placeholder items: blank SKU or entirely null data (llama3.2:3b quirk)
    real_new_inv = [
        i for i in new.inventory
        if i.sku and i.sku.strip()
        and (i.level_pct is not None or i.qty_estimate is not None)
    ]
    if real_new_inv:
        old_by_sku = {i.sku: i for i in (prev.inventory if prev else [])}
        merged_inv = []
        for item in real_new_inv:
            old = old_by_sku.get(item.sku)
            if old:
                # Fill in missing numeric fields from previous snapshot for this SKU
                from app.ai.schemas import InventorySignal
                merged_inv.append(InventorySignal(
                    sku=item.sku,
                    level_pct=item.level_pct if item.level_pct is not None else old.level_pct,
                    qty_estimate=item.qty_estimate if item.qty_estimate is not None else old.qty_estimate,
                    confidence=item.confidence if item.confidence is not None else old.confidence,
                ))
            else:
                merged_inv.append(item)
        merged.inventory = merged_inv
    return merged


def _num_hi(n: int) -> str:
    """Convert an integer to Hindi words (Indian numbering system)."""
    if n == 0:
        return "शून्य"
    if n < 0:
        return "माइनस " + _num_hi(-n)

    _ones = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
             "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह",
             "सत्रह", "अठारह", "उन्नीस"]
    _tens = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]

    def _below_hundred(x):
        if x < 20:
            return _ones[x]
        t, o = divmod(x, 10)
        return (_tens[t] + (" " + _ones[o] if o else "")).strip()

    def _below_thousand(x):
        if x < 100:
            return _below_hundred(x)
        h, r = divmod(x, 100)
        return (_ones[h] + " सौ" + (" " + _below_hundred(r) if r else "")).strip()

    parts = []
    if n >= 10_000_000:
        c, n = divmod(n, 10_000_000)
        parts.append(_below_thousand(c) + " करोड़")
    if n >= 100_000:
        l, n = divmod(n, 100_000)
        parts.append(_below_thousand(l) + " लाख")
    if n >= 1_000:
        t, n = divmod(n, 1_000)
        parts.append(_below_thousand(t) + " हज़ार")
    if n > 0:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def _recommendation_text(state: BusinessState, risk) -> str:
    """
    Returns a tuple: (display_text, tts_text)
    display_text — readable Hinglish for the frontend UI
    tts_text     — clean Devanagari Hindi for MMS TTS (no Roman, no symbols)
    """
    inv_levels = [i.level_pct for i in state.inventory if i.level_pct is not None]

    # ── UI text (Hinglish, symbols OK) ────────────────────────────────────────
    if inv_levels:
        avg = sum(inv_levels) / len(inv_levels)
        if avg < 40:
            inv_ui = f"Stock low hai (~{avg:.0f}%). Aaj reorder karein."
        else:
            inv_ui = f"Stock theek hai (~{avg:.0f}%)."
    else:
        inv_ui = "Stock update nahi mila, ek baar warehouse check karein."

    pay_ui = ""
    if (state.credit_outstanding_inr or 0) > 0 and (state.payment_due_days or 0) > 0:
        pay_ui = (f"Payment ₹{state.credit_outstanding_inr:.0f} "
                  f"— {state.payment_due_days} din se pending.")

    risk_ui = (f"Stockout risk {risk.stockout_risk:.0%}, "
               f"payment delay risk {risk.payment_delay_risk:.0%}.")
    display = f"Sir, {inv_ui} {pay_ui} {risk_ui}".strip()

    # ── TTS text (pure Devanagari, numbers as Hindi words) ───────────────────
    if inv_levels:
        avg = sum(inv_levels) / len(inv_levels)
        avg_words = _num_hi(int(round(avg)))
        if avg < 40:
            inv_tts = f"स्टॉक बहुत कम है, लगभग {avg_words} प्रतिशत। आज ही रीऑर्डर करें।"
        else:
            inv_tts = f"स्टॉक ठीक है, लगभग {avg_words} प्रतिशत।"
    else:
        inv_tts = "स्टॉक की जानकारी नहीं मिली, एक बार गोदाम चेक करें।"

    pay_tts = ""
    if (state.credit_outstanding_inr or 0) > 0 and (state.payment_due_days or 0) > 0:
        amt_words  = _num_hi(int(state.credit_outstanding_inr))
        days_words = _num_hi(int(state.payment_due_days))
        pay_tts = f"पेमेंट {amt_words} रुपये, {days_words} दिन से बाकी है।"

    so_pct  = int(round(risk.stockout_risk * 100))
    pd_pct  = int(round(risk.payment_delay_risk * 100))
    so_words = _num_hi(so_pct)
    pd_words = _num_hi(pd_pct)
    risk_tts = f"स्टॉकआउट का खतरा {so_words} प्रतिशत, पेमेंट देरी का खतरा {pd_words} प्रतिशत।"

    tts = f"सर, {inv_tts} {pay_tts} {risk_tts}".strip()

    # Store tts text on the result so the orchestrator can use it separately
    _recommendation_text._last_tts = tts
    return display


# Initialise the attribute so it always exists
_recommendation_text._last_tts = ""


def _inv_from_dict(d: dict):
    from app.ai.schemas import InventorySignal

    return InventorySignal(**d)

