from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TwilioWhatsAppWebhook:
    from_number: str
    num_media: int
    media_url_0: str | None
    media_content_type_0: str | None
    body: str | None
    # Internal override: set by demo routes so the orchestrator skips the HTTP fetch.
    _audio_bytes_override: bytes | None = field(default=None, repr=False)

    @property
    def client_id(self) -> str:
        # Twilio format: "whatsapp:+91xxxxxxxxxx" (or "+91..." or plain demo id)
        return self.from_number.replace("whatsapp:", "").strip()

    @property
    def has_audio(self) -> bool:
        if self._audio_bytes_override is not None:
            return True
        return bool(self.media_url_0) and (self.media_content_type_0 or "").startswith("audio/")
