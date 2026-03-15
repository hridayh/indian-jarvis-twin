from __future__ import annotations

from dataclasses import dataclass

from app.ai.schemas import BusinessState, OrchestrationResult
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

    def __post_init__(self):
        self.risk_predictor = RiskPredictor()

    async def process_whatsapp_webhook(self, webhook: TwilioWhatsAppWebhook) -> OrchestrationResult:
        self.state_store.ensure_client(webhook.client_id)

        if webhook.has_audio and webhook.media_url_0:
            # NOTE: Twilio media URLs often require auth. This scaffold does unauthenticated fetch;
            # wire Basic Auth (AccountSid/AuthToken) here if needed.
            audio_bytes = download_bytes(webhook.media_url_0)
            stt_out = self.stt.transcribe(audio_bytes, content_type=webhook.media_content_type_0)
            transcript = stt_out["text"]
            self.state_store.append_event(
                webhook.client_id,
                event_type="whatsapp_voice_note",
                payload={"transcript": transcript, "stt": stt_out},
            )
        else:
            transcript = (webhook.body or "").strip()
            self.state_store.append_event(
                webhook.client_id,
                event_type="whatsapp_text",
                payload={"text": transcript},
            )

        vision_payload = None
        state = self.extractor.extract(transcript=transcript, vision_payload=vision_payload)

        merged = _merge_state(self.state_store.get_latest_state(webhook.client_id), state)
        self.state_store.set_latest_state(webhook.client_id, merged)

        recent_events = self.state_store.get_recent_events(webhook.client_id, limit=200)
        risk = self.risk_predictor.predict(merged, recent_events)
        recommendation = _recommendation_text(merged, risk)

        tts_b64 = None
        try:
            wav = self.tts.synthesize_wav(recommendation, language="hi")
            tts_b64 = self.tts.wav_bytes_to_b64(wav)
        except Exception:
            # TTS is optional; don't fail ingestion.
            tts_b64 = None

        return OrchestrationResult(
            client_id=webhook.client_id,
            recommendation_text=recommendation,
            risk={"stockout": risk.stockout_risk, "payment_delay": risk.payment_delay_risk},
            updated_state=merged,
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

        tts_b64 = None
        try:
            wav = self.tts.synthesize_wav(recommendation, language="hi")
            tts_b64 = self.tts.wav_bytes_to_b64(wav)
        except Exception:
            tts_b64 = None

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

    # Inventory
    if new.inventory:
        merged.inventory = new.inventory
    return merged


def _recommendation_text(state: BusinessState, risk) -> str:
    # Hinglish / Hindi leaning recommendation scaffold
    inv_levels = [i.level_pct for i in state.inventory if i.level_pct is not None]
    inv_msg = ""
    if inv_levels:
        avg = sum(inv_levels) / len(inv_levels)
        if avg < 40:
            inv_msg = f"Stock low lag raha hai (~{avg:.0f}%). Aaj hi reorder kar dijiye."
        else:
            inv_msg = f"Stock theek hai (~{avg:.0f}%)."
    else:
        inv_msg = "Stock ka exact level clear nahi hai, par update mil gaya."

    pay_msg = ""
    if (state.credit_outstanding_inr or 0) > 0 and (state.payment_due_days or 0) > 0:
        pay_msg = f"Payment follow-up: ₹{state.credit_outstanding_inr:.0f} due, {state.payment_due_days} din se pending."

    risk_msg = f"Risk: stockout {risk.stockout_risk:.0%}, payment delay {risk.payment_delay_risk:.0%}."
    return f"Sir, {inv_msg} {pay_msg} {risk_msg}".strip()


def _inv_from_dict(d: dict):
    from app.ai.schemas import InventorySignal

    return InventorySignal(**d)

