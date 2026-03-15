from __future__ import annotations

import base64
import io
import tempfile


class CoquiTTS:
    """
    OSS TTS via Coqui TTS.

    This is heavy; if you want a lighter default, swap this implementation.
    Returns WAV bytes.
    """

    def __init__(self, *, model_name: str):
        self.model_name = model_name
        self._tts = None

    def _lazy_tts(self):
        if self._tts is not None:
            return self._tts
        try:
            from TTS.api import TTS
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Coqui TTS not installed/working.") from e
        self._tts = TTS(self.model_name)
        return self._tts

    def synthesize_wav(self, text: str, *, speaker_wav: str | None = None, language: str | None = None) -> bytes:
        tts = self._lazy_tts()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as f:
            # XTTS supports language codes; if not supported by model, it will ignore.
            kwargs = {}
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            if language:
                kwargs["language"] = language
            tts.tts_to_file(text=text, file_path=f.name, **kwargs)
            f.seek(0)
            return f.read()

    @staticmethod
    def wav_bytes_to_b64(wav_bytes: bytes) -> str:
        return base64.b64encode(wav_bytes).decode("ascii")

