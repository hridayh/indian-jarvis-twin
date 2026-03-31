from __future__ import annotations

import io
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class WhisperSTT:
    """
    Multilingual STT wrapper (faster-whisper).
    Works for Hinglish by default due to multilingual training; language can be inferred.
    """

    def __init__(self, *, model_name: str = "medium", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper not installed/working. Install deps or swap STT implementation."
            ) from e
        # Prefer CUDA if requested; fall back to CPU if CUDA init fails (common on misconfigured setups)
        try:
            logger.info("[STT] Loading faster-whisper model '%s' on device='%s'", self.model_name, self.device)
            self._model = WhisperModel(self.model_name, device=self.device)
            logger.info("[STT] Model loaded on %s", self.device)
        except Exception as e:
            logger.warning("[STT] CUDA init failed (%s), falling back to CPU", e)
            self._model = WhisperModel(self.model_name, device="cpu")
            logger.info("[STT] Model loaded on cpu (fallback)")
        return self._model

    def transcribe(self, audio_bytes: bytes, *, content_type: str | None = None) -> dict:
        """
        Returns dict: {text, segments, language}
        """
        # faster-whisper expects a file path; write to temp.
        suffix = _guess_suffix(content_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            logger.info("[STT] Starting transcription (audio size=%d bytes)", len(audio_bytes))
            try:
                model = self._lazy_model()
                segments, info = model.transcribe(tmp_path, beam_size=5)
            except RuntimeError as cuda_err:
                # CUDA libs missing at inference time — reload on CPU and retry
                logger.warning("[STT] CUDA inference failed (%s), retrying on CPU", cuda_err)
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_name, device="cpu")
                model = self._model
                segments, info = model.transcribe(tmp_path, beam_size=5)
            text = "".join(seg.text for seg in segments).strip()
            lang = getattr(info, "language", None)
            logger.info("[STT] ✅ Transcript [lang=%s]: %r", lang, text)
            return {
                "text": text,
                "language": lang,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in segments
                ],
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _guess_suffix(content_type: str | None) -> str:
    if not content_type:
        return ".bin"
    if "mpeg" in content_type:
        return ".mp3"
    if "wav" in content_type:
        return ".wav"
    if "ogg" in content_type:
        return ".ogg"
    if "mp4" in content_type or "m4a" in content_type:
        return ".m4a"
    return ".bin"

