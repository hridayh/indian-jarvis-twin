from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.ingestion.twilio_whatsapp import TwilioWhatsAppWebhook

router = APIRouter()


@router.post("/ingest/whatsapp/voice-note")
async def ingest_whatsapp_voice_note(
    request: Request,
    From: str = Form(...),
    NumMedia: int = Form(0),
    MediaUrl0: str | None = Form(None),
    MediaContentType0: str | None = Form(None),
    Body: str | None = Form(None),
) -> dict:
    """
    Twilio WhatsApp webhook receiver.

    - For voice notes: Twilio posts `MediaUrl0` pointing to audio (requires auth to fetch).
    - For text: Twilio posts `Body`.
    """
    orchestrator = request.app.state.orchestrator

    webhook = TwilioWhatsAppWebhook(
        from_number=From,
        num_media=NumMedia,
        media_url_0=MediaUrl0,
        media_content_type_0=MediaContentType0,
        body=Body,
    )

    result = await orchestrator.process_whatsapp_webhook(webhook)
    return result.model_dump()


@router.post("/ingest/cctv/snapshot")
async def ingest_cctv_snapshot(
    request: Request,
    client_id: str = Form(...),
    camera_id: str | None = Form(None),
    image: UploadFile = File(...),
) -> dict:
    orchestrator = request.app.state.orchestrator
    image_bytes = await image.read()

    result = await orchestrator.process_cctv_snapshot(
        client_id=client_id,
        camera_id=camera_id,
        image_bytes=image_bytes,
        content_type=image.content_type,
    )
    return result.model_dump()

