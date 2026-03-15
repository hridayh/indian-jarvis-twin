from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwilioWhatsAppWebhook:
    from_number: str
    num_media: int
    media_url_0: str | None
    media_content_type_0: str | None
    body: str | None

    @property
    def client_id(self) -> str:
        # Twilio format: "whatsapp:+91xxxxxxxxxx" (or "+91...")
        return self.from_number.replace("whatsapp:", "").strip()

    @property
    def has_audio(self) -> bool:
        return bool(self.media_url_0) and (self.media_content_type_0 or "").startswith("audio/")

