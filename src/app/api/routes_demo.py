from __future__ import annotations

"""
Demo-friendly routes — no Twilio webhook format needed.
Designed to be called directly from a browser or curl.
"""

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/demo", tags=["demo"])


@router.post("/voice")
async def demo_voice(
    request: Request,
    client_id: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    """
    Accept a raw audio file from the browser (WebM/OGG/WAV/MP3).
    Runs:  STT → LLM extraction → Digital Twin update → risk → TTS
    Returns the full OrchestrationResult as JSON.
    """
    orchestrator = request.app.state.orchestrator
    audio_bytes = await audio.read()
    logger.info("[/demo/voice] client=%s audio=%d bytes content_type=%s",
                client_id, len(audio_bytes), audio.content_type)

    # Simulate a Twilio-style webhook internally so we reuse the same
    # orchestration path, but we inject bytes directly instead of fetching
    # from a Twilio URL.
    from app.ingestion.twilio_whatsapp import TwilioWhatsAppWebhook

    webhook = TwilioWhatsAppWebhook(
        from_number=client_id,
        num_media=0,
        media_url_0=None,
        media_content_type_0=audio.content_type,
        body=None,
        # Attach bytes directly so the orchestrator skips the HTTP download.
        _audio_bytes_override=audio_bytes,
    )
    result = await orchestrator.process_whatsapp_webhook(webhook)
    return result.model_dump()


@router.post("/cctv")
async def demo_cctv(
    request: Request,
    client_id: str = Form(...),
    camera_id: str | None = Form(None),
    image: UploadFile = File(...),
) -> dict:
    """
    Accept a CCTV snapshot image from the browser.
    Runs:  Vision → Digital Twin update → risk → TTS
    """
    orchestrator = request.app.state.orchestrator
    image_bytes = await image.read()
    logger.info("[/demo/cctv] client=%s camera=%s image=%d bytes", client_id, camera_id, len(image_bytes))
    result = await orchestrator.process_cctv_snapshot(
        client_id=client_id,
        camera_id=camera_id,
        image_bytes=image_bytes,
        content_type=image.content_type,
    )
    return result.model_dump()


@router.post("/text")
async def demo_text(
    request: Request,
    client_id: str = Form(...),
    message: str = Form(...),
) -> dict:
    """
    Accept a plain text / WhatsApp text message from the browser.
    Runs:  LLM extraction → Digital Twin update → risk → TTS
    Useful for quick demos without a mic.
    """
    orchestrator = request.app.state.orchestrator
    logger.info("[/demo/text] client=%s message=%r", client_id, message[:120])

    from app.ingestion.twilio_whatsapp import TwilioWhatsAppWebhook

    webhook = TwilioWhatsAppWebhook(
        from_number=client_id,
        num_media=0,
        media_url_0=None,
        media_content_type_0=None,
        body=message,
    )
    result = await orchestrator.process_whatsapp_webhook(webhook)
    return result.model_dump()
